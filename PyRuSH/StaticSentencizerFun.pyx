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
                sentence_len = len(token.text)
                marked_this_span = True
            sentence_len = token.idx + len(token.text) - sentence_start_idx
            if max_sentence_length is not None and sentence_len > max_sentence_length:
                doc_guesses[t] = True
                logger.debug(f"[doc {doc_idx}] Mark/Split due to max_sentence_length at token {t}: '{token.text}' idx={token.idx}")
                sentence_start_t = t
                sentence_start_idx = token.idx
                sentence_len = len(token.text)
            t += 1
        logger.debug(f"[doc {doc_idx}] Sentence start guesses: {[i for i, v in enumerate(doc_guesses) if v]}")
        guesses.append(doc_guesses)
    return guesses

cpdef cpredict_split_gaps(docs, sentencizer_fun, max_sentence_length=None):
    cdef list guesses
    guesses = []
    call_id = getattr(cpredict_split_gaps, 'call_id', 0)
    setattr(cpredict_split_gaps, 'call_id', call_id + 1)
    for doc_idx, doc in enumerate(docs):
        if len(doc) == 0:
            guesses.append([])
            continue
        doc_guesses = [False] * len(doc)
        sentence_spans = sentencizer_fun(doc.text)
        num_spans = len(sentence_spans)
        t = 0
        span_idx = 0
        sentence_len = 0
        is_first_token_in_span = True
        next_span_begin = sentence_spans[span_idx + 1].begin if num_spans > 1 else -1
        while t < len(doc):
            token = doc[t]
            # Advance to next span if needed
            while span_idx < num_spans and token.idx >= sentence_spans[span_idx].end:
                span_idx += 1
                is_first_token_in_span = True
                next_span_begin = sentence_spans[span_idx + 1].begin if span_idx < num_spans - 1 else -1
            if span_idx >= num_spans:
                # After all spans, only mark whitespace tokens as sentence start
                if len(token.text.strip()) == 0:
                    doc_guesses[t] = True
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (whitespace after all spans)")
                t += 1
                continue
            span = sentence_spans[span_idx]
            # If before the span, skip
            if token.idx < span.begin:
                t += 1
                continue
            # If in the span
            if token.idx < span.end:
                if is_first_token_in_span:
                    doc_guesses[t] = True
                    is_first_token_in_span = False
                    sentence_len = len(token.text)
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (span {span_idx})")
                elif max_sentence_length is not None and sentence_len + len(token.text) > max_sentence_length:
                    doc_guesses[t] = True
                    sentence_len = len(token.text)
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (max length split in span {span_idx})")
                else:
                    sentence_len += len(token.text)
                # If we just split, don't add token to sentence_len again
                t += 1
                continue
            # After the span, before next span, mark whitespace tokens
            if next_span_begin != -1 and token.idx < next_span_begin:
                if len(token.text.strip()) == 0:
                    doc_guesses[t] = True
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (whitespace after span {span_idx})")
                    t += 1
                    continue
                else:
                    t += 1
                    continue
            # If no next span, just move on
            t += 1
        logger.debug(f'[cpredict_split_gaps|call_id={call_id}] Token/tag mapping: ' + str([(d, l) for d, l in zip(list(doc), doc_guesses)]))
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
