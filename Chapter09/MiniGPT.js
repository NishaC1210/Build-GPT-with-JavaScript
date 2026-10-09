import { Tensor } from './Tensor.js';
import { RMSNorm, TransformerBlock } from './Transformer.js';

export class MiniGPT {
    constructor(vocabSize, featureDim, ropeBase, numBlocks) {
        this.vocabSize = vocabSize;
        this.featureDim = featureDim;
        this.ropeBase = ropeBase; // replaces maxContextLength
        this.numBlocks = numBlocks;

        // Token embedding table (position now enters through rotation in attention).
        this.tokenEmbeddings = new Tensor(vocabSize, featureDim);
        this.tokenEmbeddings.initToSmallRandom();

        // Stack of transformer blocks.
        this.blocks = [];
        for (let i = 0; i < numBlocks; ++i) {
            this.blocks.push(new TransformerBlock(featureDim, ropeBase)); // base handed down
        }

        // Final norm before the unembedding.
        this.finalNorm = new RMSNorm();
    }

    embed(ids) {
        // Token embeddings only; position now enters through rotation in attention
        return this.tokenEmbeddings.gatherRows(ids);
    }

    forward(tokenIdArray) {
        // No maxContextLength check: there is no position table to index past.
        // A sequence beyond the trained length just runs slower and predicts worse.
        let x = this.embed(tokenIdArray); // Shape: sequence_length, feature_dim
        for (let i = 0; i < this.blocks.length; ++i) {
            x = this.blocks[i].forward(x); // Shape preserved.
        }
        x = this.finalNorm.forward(x); // Shape: sequence_length, feature_dim

        // Tied unembedding: (sequence_length, feature_dim) * (feature_dim, vocabSize)
        const logits = x.matmul(this.tokenEmbeddings.transposed()); // Shape: sequence_length, vocabSize
        return logits;
    }

    // Greedy next-token prediction: argmax over the last row's distribution.
    predictNextToken(tokenIdArray) {
        const logits = this.forward(tokenIdArray);

        // Softmax isn't needed for argmax (it's monotonic), but it's kept here
        // so this same path supports temperature sampling in a later chapter.
        const probabilities = logits.softMaxedRows();

        // Only the last row matters — it predicts the token after the sequence.
        // get now returns a plain number, so the trailing .data is gone.
        const lastRow = tokenIdArray.length - 1;
        let bestId = 0;
        let bestScore = probabilities.get(lastRow, 0);
        for (let col = 1; col < this.vocabSize; ++col) {
            const score = probabilities.get(lastRow, col);
            if (score > bestScore) {
                bestScore = score;
                bestId = col;
            }
        }
        return bestId;
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

        // Final norm has no parameters yet, but ask anyway for forward-compatibility.
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
