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
        spans = sentencizer_fun(doc.text)
        logger.debug(f"[doc {doc_idx}] {len(spans)} spans detected: {[ (span.begin, span.end) for span in spans ]}")
        t = 0
        span_idx = 0
        num_spans = len(spans)
        sentence_start_t = 0
        last_ws_token = None
        while t < len(doc):
            token = doc[t]
            # 1. Mark token as sentence start if it overlaps with RuSH span.begin
            if span_idx < num_spans and token.idx == spans[span_idx].begin:
                doc_guesses[t] = True
                logger.debug(f"[doc {doc_idx}] Mark sentence start at token {t}: '{token.text}' idx={token.idx} (span begin)")
                sentence_start_t = t
                last_ws_token = None
                span = spans[span_idx]
                # Find last token in span
                last_token_in_span = t
                while last_token_in_span + 1 < len(doc) and doc[last_token_in_span + 1].idx < span.end:
                    last_token_in_span += 1
                if last_token_in_span + 1 < len(doc) and doc[last_token_in_span + 1].idx >= span.end:
                    # Mark whitespace preferred, else next token
                    if doc[last_token_in_span + 1].text.isspace():
                        doc_guesses[last_token_in_span + 1] = True
                        logger.debug(f"[doc {doc_idx}] Token {last_token_in_span+1} '{doc[last_token_in_span+1].text}' marked as sentence start (span end whitespace)")
                    else:
                        doc_guesses[last_token_in_span + 1] = True
                        logger.debug(f"[doc {doc_idx}] Token {last_token_in_span+1} '{doc[last_token_in_span+1].text}' marked as sentence start (span end next token)")
                span_idx += 1
                t += 1
                continue
            # 2. If token is in gap between spans
            if span_idx > 0 and token.idx >= spans[span_idx-1].end and (span_idx < num_spans and token.idx < spans[span_idx].begin):
                gap_start = t
                gap_end = t
                while gap_end < len(doc) and doc[gap_end].idx < spans[span_idx].begin:
                    gap_end += 1
                whitespace_found = False
                for i in range(gap_start, gap_end):
                    if doc[i].text.isspace():
                        doc_guesses[i] = True
                        logger.debug(f"[doc {doc_idx}] Mark sentence start at token {i}: '{doc[i].text}' idx={doc[i].idx} (gap whitespace)")
                        whitespace_found = True
                        if i+1 < gap_end and not doc[i+1].text.isspace():
                            doc_guesses[i+1] = True
                            logger.debug(f"[doc {doc_idx}] Mark sentence start at token {i+1}: '{doc[i+1].text}' idx={doc[i+1].idx} (gap non-whitespace after whitespace)")
                        break
                if not whitespace_found:
                    for i in range(gap_start, gap_end):
                        if not doc[i].text.isspace():
                            doc_guesses[i] = True
                            logger.debug(f"[doc {doc_idx}] Mark sentence start at token {i}: '{doc[i].text}' idx={doc[i].idx} (gap non-whitespace)")
                            break
                t = gap_end
                continue
            # 3. Split at last whitespace or previous token BEFORE exceeding max length
            if max_sentence_length is not None:
                sentence_len = 0
                last_ws_token = -1
                for k in range(sentence_start_t, t):
                    sentence_len += len(doc[k].text)
                    if doc[k].text.isspace():
                        last_ws_token = k
                current_token_len = len(token.text)
                # If adding current token would exceed max length
                if sentence_len + current_token_len > max_sentence_length:
                    # Find split point: last whitespace before limit, else previous token
                    split_token = last_ws_token if last_ws_token >= sentence_start_t else t-1 if t > sentence_start_t else t
                    # Prevent split_token from being the same as sentence_start_t (in case no whitespace and only one token)
                    if split_token == sentence_start_t and t > sentence_start_t:
                        split_token = t-1
                    doc_guesses[split_token] = True
                    logger.debug(f"[doc {doc_idx}] Mark/Split due to max_sentence_length at token {split_token}: '{doc[split_token].text}' idx={doc[split_token].idx} (split before exceeding limit)")
                    sentence_start_t = split_token
                    continue
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
        sentence_start_idx = 0
        is_first_token_in_span = True
        while t < len(doc):
            token = doc[t]
            # Always check for gaps between spans before advancing span_idx
            next_span_begin = sentence_spans[span_idx + 1].begin if span_idx < num_spans - 1 else -1
            # 1. Handle gaps between spans
            if span_idx < num_spans - 1 and token.idx >= sentence_spans[span_idx].end and token.idx < next_span_begin:
                gap_start = t
                gap_end = t
                while gap_end < len(doc) and doc[gap_end].idx < next_span_begin:
                    gap_end += 1
                logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] GAP DETECTED: tokens {gap_start}-{gap_end-1} (idx {doc[gap_start].idx}-{doc[gap_end-1].idx}) between spans {sentence_spans[span_idx].end}-{next_span_begin}")
                # Mark first whitespace token in gap, else first token
                whitespace_idx = -1
                for i in range(gap_start, gap_end):
                    if doc[i].text.isspace():
                        whitespace_idx = i
                        break
                if whitespace_idx != -1:
                    doc_guesses[whitespace_idx] = True
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {whitespace_idx} '{doc[whitespace_idx].text}' marked as sentence start (whitespace in gap between spans)")
                else:
                    doc_guesses[gap_start] = True
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {gap_start} '{doc[gap_start].text}' marked as sentence start (first token in gap between spans)")
                t = gap_end
                continue
            # 2. Advance span_idx if needed
            while span_idx < num_spans and token.idx >= sentence_spans[span_idx].end:
                span_idx += 1
            if span_idx >= num_spans:
                # After all spans, only mark whitespace tokens as sentence start
                if token.text.isspace():
                    doc_guesses[t] = True
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (whitespace after all spans)")
                t += 1
                continue
            span = sentence_spans[span_idx]
            # 3. If before the span, skip
            if token.idx < span.begin:
                t += 1
                continue
            # 4. If in the span
            if token.idx < span.end:
                # Mark sentence start if token overlaps with span.begin
                if token.idx == span.begin:
                    doc_guesses[t] = True
                    sentence_start_idx = token.idx
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (span begin)")
                # If sentence length exceeds max_sentence_length, mark as sentence start
                elif max_sentence_length is not None and (token.idx - sentence_start_idx) + len(token.text) > max_sentence_length:
                    doc_guesses[t] = True
                    sentence_start_idx = token.idx
                    logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t} '{token.text}' marked as sentence start (max length split in span {span_idx})")
                # If this is the last token in the span, mark next token as sentence start (if exists)
                if t + 1 < len(doc) and doc[t + 1].idx >= span.end:
                    # Mark whitespace preferred, else next token
                    if doc[t + 1].text.isspace():
                        doc_guesses[t + 1] = True
                        logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t+1} '{doc[t+1].text}' marked as sentence start (span end whitespace)")
                    else:
                        doc_guesses[t + 1] = True
                        logger.debug(f"[cpredict_split_gaps|call_id={call_id}] [doc {doc_idx}] Token {t+1} '{doc[t+1].text}' marked as sentence start (span end next token)")
                t += 1
                continue
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
