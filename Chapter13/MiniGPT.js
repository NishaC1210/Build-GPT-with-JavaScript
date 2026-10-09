import { Tensor } from './Tensor.js';
import { RMSNorm, TransformerBlock } from './Transformer.js';

export class MiniGPT {
    constructor(vocabSize, featureDim, numHeads, ropeBase, numBlocks) {
        this.vocabSize = vocabSize;
        this.featureDim = featureDim;
        this.numHeads = numHeads;
        this.ropeBase = ropeBase; // replaces maxContextLength
        this.numBlocks = numBlocks;

        // Token embedding table (position now enters through rotation in attention).
        this.tokenEmbeddings = new Tensor(vocabSize, featureDim);
        this.tokenEmbeddings.initToSmallRandom();

        // Stack of transformer blocks.
        this.blocks = [];
        for (let i = 0; i < numBlocks; ++i) {
            this.blocks.push(new TransformerBlock(featureDim, numHeads, ropeBase)); // heads and base handed down
        }

        // Final norm before the unembedding.
        this.finalNorm = new RMSNorm(featureDim);
    }

    embed(ids) {
        // Token embeddings only; position now enters through rotation in attention
        return this.tokenEmbeddings.gatherRows(ids);
    }

    forward(tokenIdArray, cache = null) { // cache is new
        // No maxContextLength check: there is no position table to index past.
        // A sequence beyond the trained length just runs slower and predicts worse.
        // With a cache, tokenIdArray holds only the new tokens; the block index
        // threads through so each layer reads and writes its own cached slot.
        let x = this.embed(tokenIdArray); // Shape: new_tokens, feature_dim
        for (let i = 0; i < this.blocks.length; ++i) {
            x = this.blocks[i].forward(x, cache, i); // Shape preserved.
        }
        x = this.finalNorm.forward(x); // Shape: new_tokens, feature_dim

        // Tied unembedding: (sequence_length, feature_dim) * (feature_dim, vocabSize)
        const logits = x.matmul(this.tokenEmbeddings.transposed()); // Shape: sequence_length, vocabSize
        return logits;
    }

    // Predict the next token. Temperature reshapes the distribution, top-k and
    // top-p rule out the unlikely tail, and an optional KV cache skips rerunning
    // tokens that were processed on an earlier call.
    predictNextToken(tokenIdArray, temperature = 1.0, topK = 50, topP = 0.9, cache = null) {
        // With a cache, the stored tokens do not run again. Only the tokens
        // past the cached length are new.
        const newTokens = cache === null
            ? tokenIdArray
            : tokenIdArray.slice(cache.length);

        const logits = this.forward(newTokens, cache); // was forward(tokenIdArray)

        // The last row holds the scores for the token after the sequence,
        // whether that is the last row of a full prompt or the only row of a
        // single cached step.
        const lastRow = logits.rows - 1;

        // Temperature zero means greedy decoding: the single highest logit.
        // Softmax is monotonic, so the argmax of the logits is the argmax of
        // the probabilities, and the softmax can be skipped here.
        if (temperature === 0) {
            let bestId = 0;
            let bestScore = logits.get(lastRow, 0);
            for (let col = 1; col < this.vocabSize; ++col) {
                const score = logits.get(lastRow, col);
                if (score > bestScore) {
                    bestScore = score;
                    bestId = col;
                }
            }
            return bestId;
        }

        // Reshape by temperature and turn the last row into probabilities.
        const scaled = logits.scale(1 / temperature);
        const probabilities = scaled.softMaxedRows();

        // Rank the tokens from most to least likely so the worst can be cut.
        const ranked = [];
        for (let col = 0; col < this.vocabSize; ++col) {
            ranked.push({ id: col, probability: probabilities.get(lastRow, col) });
        }
        ranked.sort((a, b) => b.probability - a.probability);

        // top-k: keep only the k most likely tokens.
        const capped = ranked.slice(0, topK);

        // top-p: from the top, keep tokens until their combined probability
        // crosses topP, then stop. Always keeps at least the single top token.
        const kept = [];
        let keptTotal = 0.0;
        for (let i = 0; i < capped.length; ++i) {
            kept.push(capped[i]);
            keptTotal += capped[i].probability;
            if (keptTotal >= topP) {
                break;
            }
        }

        // The survivors cover only part of the number line now, up to keptTotal.
        // Draw a random number up to that total, then walk them as before.
        const draw = Math.random() * keptTotal;
        let cumulative = 0.0;
        for (let i = 0; i < kept.length; ++i) {
            cumulative += kept[i].probability;
            if (draw < cumulative) {
                return kept[i].id;
            }
        }

        return kept[kept.length - 1].id; // guard against floating-point drift
    }

    parameters() {
        const params = [];

        // Token embedding table (also used tied as the unembedding).
        const tokenParams = this.tokenEmbeddings.parameters();
        for (let i = 0; i < tokenParams.length; ++i) {
            params.push(tokenParams[i]);
        }

        // Per-block parameters.
        for (let i = 0; i < this.blocks.length; ++i) {
            const blockParams = this.blocks[i].parameters();
            for (let j = 0; j < blockParams.length; ++j) {
                params.push(blockParams[j]);
            }
        }

        // Final norm now carries a learned scale (gamma), gathered like the rest.
        const normParams = this.finalNorm.parameters();
        for (let i = 0; i < normParams.length; ++i) {
            params.push(normParams[i]);
        }

        return params;
    }

    serializeToArrayBuffer() {
        const params = this.parameters();

        // Each parameter is a tensor with many values, so size the buffer to the
        // total number of values across all tensors.
        let total = 0;
        for (const t of params) {
            total += t.data.length;
        }

        const view = new Float32Array(total);
        let offset = 0;
        for (const t of params) {
            view.set(t.data, offset);
            offset += t.data.length;
        }
        return view.buffer;
    }

    deserializeFromArrayBuffer(buffer) {
        const params = this.parameters();
        const view = new Float32Array(buffer);
        let offset = 0;
        for (const t of params) {
            t.data.set(view.subarray(offset, offset + t.data.length));
            offset += t.data.length;
        }
    }
}
