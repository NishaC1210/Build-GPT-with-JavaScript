class Tokenizer {
    constructor() {
        // Merge rules array. 
        // Index = token id. 
        // Value = the two token ids that merge into index.
        this.merges = new Array(); 

        // Reserved tokens, an array of strings
        this.reserved = new Array();

        // Seed first 256 tokens as placeholders.
        // These merge rules are not valid merges.
        for (let i = 0; i < 256; ++i) {
            this.merges.push([i, i]); 
        }
    }

    vocabSize()  { 
        return this.merges.length;
    }

    encode(text) { // string -> Array<int>
        let bytes = new TextEncoder().encode(text); 

        // Apply every merge rule, in order
        for (let rule = 256; rule < this.merges.length; ++rule) {
            bytes = this._merge(bytes, this.merges[rule][0], this.merges[rule][1], rule);
        }

        return bytes;
    }

    // Given a list of tokens, replace every consequtive instance of firstToken and secondToken with repalcementToken
    _merge(listOfTokens, firstToken, secondToken, replacementToken) {
        const result = new Array();

        for (let i = 0; i < listOfTokens.length; ++i) {
            if (listOfTokens[i] == firstToken) {
                if (i + 1 < listOfTokens.length) {
                    if (listOfTokens[i + 1] == secondToken) {
                        result.push(replacementToken);
                        i += 1;
                        continue;
                    }
                }
            }
            result.push(listOfTokens[i]);
        }

        return result;
    }

    decode(ids) { // Array<int> -> string
        const bytes = [];

        for (let i = 0; i < ids.length; ++i) {
            const stack = [ids[i]];

            while (stack.length > 0) {
                const id = stack.pop();

                if (id < 256) {
                    bytes.push(id);
                }
                else {
                    const pair = this.merges[id];
                    stack.push(pair[1]); // second half pushed first,
                    stack.push(pair[0]); // so first half pops first
                }
            }
        }

        return new TextDecoder().decode(new Uint8Array(bytes));
    }

    reserve(text) { // string -> id, call before train
        let bytes = this.encode(text);

        let id = bytes[0];
        for (let i = 1; i < bytes.length; ++i) {
            this.merges.push([id, bytes[i]]);
            id = this.merges.length - 1;
        }

        this.reserved.push(text);
        return id;
    }

    // Splits long text into an array of text chunks
    _split(inputText) { // "Long text" -> ["Long", " text"]
        // Sort this.r eserved by reverse length, so _matchKeyword can match longest first
        this.reserved.sort((a, b) => b.length - a.length);

        const _matchKeyword = (text, position, keywords) => {
            for (const keyword of keywords) {
                if (text.startsWith(keyword, position)) {
                    return keyword;
                }
            }
            return null;
        }

        const _isLetter = (char) => {
            return char !== undefined && char.toLowerCase() !== char.toUpperCase();
        }

        const _isDigit = (char) => {
            return char >= '0' && char <= '9';
        }

        const chunks = [];
        let i = 0;

        while (i < inputText.length) {
            // Rule 1: keywords always win, they are atomic
            const keyword = _matchKeyword(inputText, i, this.reserved);
            if (keyword !== null) {
                chunks.push(keyword);
                i += keyword.length;
                continue;
            }

            const char = inputText[i];

            // Rule 2: a word, optionally carrying ONE leading space
            if (_isLetter(char) || (char === ' ' && _isLetter(inputText[i + 1]))) {
                let chunk = char;
                i++;
                while (i < inputText.length && _isLetter(inputText[i])) {
                    chunk += inputText[i];
                    i++;
                }
                chunks.push(chunk);
                continue;
            }

            // Rule 3: digits, grouped to at most 3
            if (_isDigit(char)) {
                let chunk = '';
                while (chunk.length < 3 && _isDigit(inputText[i])) {
                    chunk += inputText[i];
                    i++;
                }
                chunks.push(chunk);
                continue;
            }

            // Rule 4: anything else (punctuation, leftover spaces) is its own chunk
            chunks.push(char);
            i++;
        }

        return chunks;
    } 

    train(text, targetVocabSize) { // learns merges
        const splitText = this._split(text);
        const chunks = new Array(splitText.length);
        for (let i = 0; i < splitText.length; ++i) {
            chunks[i] = this.encode(splitText[i]);
        }

        while (this.merges.length < targetVocabSize) {
            const pairs = new Map();

            // Count pairs in all chunks
            for (let i = 0; i < chunks.length; ++i) {
                const chunk = chunks[i];
                for (let j = 0; j < chunk.length - 1; ++j) {
                    const key = chunk[j] + ',' + chunk[j + 1];

                    const entry = pairs.get(key);
                    if (entry) {
                        entry.count += 1;
                    }
                    else {
                        pairs.set(key, { count: 1, firstToken: chunk[j], secondToken: chunk[j + 1] });
                    }
                }
            }

            // Pick Best Pair
            let best = null;
            for (const entry of pairs.values()) {
                if (best === null || entry.count > best.count) {
                    best = entry;
                }
            }

            // Early out
            if (best == null) {
                break; // No best pair, all merged
            }
            else if (best.count < 2) {
                // If the best token was only seen once
                break; 
            }

            // Record new rule
            this.merges.push([best.firstToken, best.secondToken]);
            const newToken = this.merges.length - 1;

            // Apply new rule to each chunk
            for (let i = 0; i < chunks.length; ++i) {
                chunks[i] = this._merge(chunks[i], best.firstToken, best.secondToken, newToken);
            }
        }
    }    

    serializeToJSON() { // -> string
        return JSON.stringify({
            reserved: this.reserved,
            merges: this.merges.slice(256), // ids 0-255 are seeded, never saved
        });
    }

    deserializeFromJSON(json)  { 
        const data = JSON.parse(json);

        // Restore the freshly-constructed state: 256 byte tokens, nothing else.
        this.merges.length = 256;
        this.reserved = data.reserved;

        // Pushing the pairs back in order reproduces every id exactly
        for (let i = 0; i < data.merges.length; ++i) {
            this.merges.push(data.merges[i]);
        }
    }
}