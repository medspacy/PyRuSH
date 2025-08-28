from loguru import logger
# ******************************************************************************
#  MIT License
#
#  Copyright (c) 2020 Jianlin Shi
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
#  modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ******************************************************************************
cpdef cpredict_merge_gaps(docs, sentencizer_fun, max_sentence_length=None):
    cdef list guesses
    guesses = []
    logger.debug(f"cpredict_merge_gaps called: docs={len(docs)}, max_sentence_length={max_sentence_length}")
    for doc_idx, doc in enumerate(docs):
        if len(doc) == 0:
            guesses.append([])
            continue
        doc_guesses = [False] * len(doc)
        orig_spans = sentencizer_fun(doc.text)
        logger.debug(f"[doc {doc_idx}] {len(orig_spans)} spans detected: {[ (span.begin, span.end) for span in orig_spans ]}")
        t = 0
        s = 0
        sentence_start_t = None
        sentence_start_idx = None
        sentence_len = 0
        marked_this_span = False
        while t < len(doc):
            token = doc[t]
            # Advance to next span if needed
            while s < len(orig_spans) and token.idx >= orig_spans[s].end:
                s += 1
                marked_this_span = False
            if s >= len(orig_spans):
                break
            span = orig_spans[s]
            # Only process tokens within the span
            if token.idx < span.begin or token.idx >= span.end:
                t += 1
                continue
            if len(token.text.strip()) == 0:
                t += 1
                continue
            # Mark the first non-whitespace token of the span as sentence start
            if not marked_this_span:
                doc_guesses[t] = True
                logger.debug(f"[doc {doc_idx}] Mark sentence start at token {t}: '{token.text}' idx={token.idx} (span start)")
                sentence_start_t = t
                sentence_start_idx = token.idx
                sentence_len = 0
                marked_this_span = True
            sentence_len = token.idx + len(token.text) - sentence_start_idx
            if max_sentence_length is not None and sentence_len > max_sentence_length:
                doc_guesses[t] = True
                logger.debug(f"[doc {doc_idx}] Split due to max_sentence_length at token {t}: '{token.text}' idx={token.idx}")
                sentence_start_t = t
                sentence_start_idx = token.idx
                sentence_len = 0
            t += 1
        logger.debug(f"[doc {doc_idx}] Sentence start guesses: {[i for i, v in enumerate(doc_guesses) if v]}")
        guesses.append(doc_guesses)
    return guesses

cpdef cpredict_split_gaps(docs, sentencizer_fun, max_sentence_length=None):
    cdef list guesses
    cdef int s
    cdef int t
    cdef int last_span_end
    guesses = []
    for doc_idx, doc in enumerate(docs):
        if len(doc) == 0:
            guesses.append([])
            continue
        doc_guesses = [False] * len(doc)
        sentence_spans = sentencizer_fun(doc.text)
        s = 0
        t = 0
        last_span_end = -1  # Track the end of the last span
        prev_span_end = None
        sentence_start_t = None
        sentence_start_idx = None
        sentence_len = 0
        marked_this_span = False
        while t < len(doc):
            token = doc[t]
            # Check for gap between previous span and current span
            if s < len(sentence_spans):
                span = sentence_spans[s]
                span_begin = span[0]
                span_end = span[1]
                # If there is a gap between previous span and current span
                if prev_span_end is not None and span_begin >= prev_span_end:
                    # Always mark the first token after prev_span_end, even if whitespace
                    for gap_t in range(t, len(doc)):
                        gap_token = doc[gap_t]
                        if gap_token.idx >= prev_span_end:
                            doc_guesses[gap_t] = True
                            t = gap_t
                            # Reset sentence tracking for new sentence
                            sentence_start_t = gap_t
                            sentence_start_idx = gap_token.idx
                            sentence_len = 0
                            break
                    prev_span_end = None
                    continue
                # Mark the first token of the span
                if token.idx <= span_begin < token.idx + len(token):
                    doc_guesses[t] = True
                    prev_span_end = span_end
                    sentence_start_t = t
                    sentence_start_idx = token.idx
                    sentence_len = 0
                    t += 1
                    s += 1
                    continue
                elif token.idx + len(token) <= span_begin:
                    t += 1
                    continue
                else:
                    prev_span_end = span_end
                    s += 1
                    continue
            else:
                # After all spans, handle any trailing tokens after last span
                if prev_span_end is not None and token.idx > prev_span_end:
                    doc_guesses[t] = True
                    prev_span_end = None
                    sentence_start_t = t
                    sentence_start_idx = token.idx
                    sentence_len = 0
                    t += 1
                    continue
            # Sentence length logic
            if sentence_start_idx is not None:
                sentence_len = token.idx + len(token.text) - sentence_start_idx
                if max_sentence_length is not None and sentence_len > max_sentence_length:
                    doc_guesses[t] = True
                    sentence_start_t = t
                    sentence_start_idx = token.idx
                    sentence_len = 0
            t += 1
        guesses.append(doc_guesses)
    return guesses

cpdef cset_annotations(docs, batch_tag_ids, tensors=None):
    if type(docs) !=list:
        docs = [docs]
    for i, doc in enumerate(docs):
        doc_tag_ids = batch_tag_ids[i]
        for j, tag_id in enumerate(doc_tag_ids):
            # Don't clobber existing sentence boundaries
            if tag_id:
                doc[j].sent_start = True
            else:
                doc[j].sent_start = False

# The function 'char_span' will try to match the tokens in the backend, as it might be less efficient when match
# sentences where it does not assume the sentences are sorted. Also, it will return None if not find a match rather
# than looking around. Thus, abandon this method.

# cpdef csegment(doc, sentencizer_fun):
#     for token in doc:
#         token.is_sent_start = False
#     sentence_spans = sentencizer_fun(doc.text)
#     for span in sentence_spans:
#         sent = doc.char_span(span.begin, span.end)
#         sent[0].is_sent_start = True
