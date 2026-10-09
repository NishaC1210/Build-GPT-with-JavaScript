import { Matrix } from './Matrix.js';
import { RMSNorm, TransformerBlock } from './Transformer.js';

export class MiniGPT {
    constructor(vocabSize, featureDim, maxContextLength, numBlocks) {
        this.vocabSize = vocabSize;
        this.featureDim = featureDim;
        this.maxContextLength = maxContextLength;
        this.numBlocks = numBlocks;

        // Embedding Table
        this.tokenEmbeddings = new Matrix(vocabSize, featureDim);
        this.tokenEmbeddings.initToSmallRandom();

        this.positionalEmbeddings = new Matrix(maxContextLength, featureDim);
        this.positionalEmbeddings.initToSmallRandom();

        // Stack of transformer blocks.
        this.blocks = [];
        for (let i = 0; i < numBlocks; ++i) {
            this.blocks.push(new TransformerBlock(featureDim));
        }

        // Final norm before the unembedding.
        this.finalNorm = new RMSNorm();
    }

    embed(tokenIdArray) { // Combine token and positional embeddings
        const sequenceLength = tokenIdArray.length;
        const embedded = new Matrix(sequenceLength, this.featureDim);

        for (let position = 0; position < sequenceLength; ++position) {
            const tokenId = tokenIdArray[position];

            for (let feature = 0; feature < this.featureDim; ++feature) {
                const tokenComponent =
                    this.tokenEmbeddings.get(tokenId, feature);

                const positionComponent =
                    this.positionalEmbeddings.get(position, feature);

                embedded.set(
                    position,
                    feature,
                    tokenComponent.add(positionComponent)
                );
            }
        }

        return embedded;
    }

    forward(tokenIdArray) {
        if (tokenIdArray.length > this.maxContextLength) {
            throw new Error(`input length ${tokenIdArray.length} exceeds maxContextLength ${this.maxContextLength}`);
        }

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

        // Only the last row matters, it predicts the token after the sequence.
        const lastRow = tokenIdArray.length - 1;
        let bestId = 0;
        let bestScore = probabilities.get(lastRow, 0).data;
        for (let col = 1; col < this.vocabSize; ++col) {
            const score = probabilities.get(lastRow, col).data;
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

        // Positional embedding table.
        const posParams = this.positionalEmbeddings.parameters();
        for (let i = 0; i < posParams.length; ++i) {
            params.push(posParams[i]);
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
        const buffer = new ArrayBuffer(params.length * 4); // Float32 = 4 bytes
        const view = new Float32Array(buffer);
        for (let i = 0; i < params.length; ++i) {
            view[i] = params[i].data;
        }
        return buffer;
    }

    deserializeFromArrayBuffer(buffer) {
        const params = this.parameters();
        const view = new Float32Array(buffer);
        if (view.length !== params.length) {
            throw new Error(`weight count mismatch: file has ${view.length}, model expects ${params.length}`);
        }
        for (let i = 0; i < params.length; ++i) {
            params[i].data = view[i];
        }
    }
}