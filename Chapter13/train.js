import { readFileSync, writeFileSync, readdirSync, openSync, writeSync, closeSync } from "node:fs";
import { join } from "node:path";
import { Tokenizer } from "./Tokenizer.js";
import { MiniGPT } from "./MiniGPT.js";
import { AdamWTrainer } from "./Trainer.js";

// The model. This is the one place its shape is set.

const MODEL_VOCAB_SIZE = 10000;
const MODEL_FEATURE_DIM = 768;
const MODEL_NUM_HEADS = 12;    // head_dim = 768 / 12 = 64, even
const MODEL_ROPE_BASE = 10000; // ample for 1024 positions
const MODEL_NUM_BLOCKS = 13;

const CONTEXT_LENGTH = 1024; // tokens per pretraining window
const BATCH_SIZE = 8;        // training examples per optimizer step
const NUM_EPOCHS = 1;

function buildModel() {
    return new MiniGPT(
        MODEL_VOCAB_SIZE,
        MODEL_FEATURE_DIM,
        MODEL_NUM_HEADS,
        MODEL_ROPE_BASE,
        MODEL_NUM_BLOCKS
    );
}

function parseArgs(argv) {
    const fields = { "-i": "input", "-o": "output", "-t": "tokenizer", "-p": "pretrained" };
    const args = {};
    let i = 0;
    while (i < argv.length) {
        const field = fields[argv[i]];
        if (field === undefined) {
            throw new Error(`unknown or misplaced argument: ${argv[i]}`);
        }
        args[field] = argv[i + 1];
        i += 2;
    }
    return args;
}

function usage() {
    console.log(
        "usage:\n" +
        "  node train.js tokenizer -i <text-folder>        -o vocab.json\n" +
        "  node train.js encode    -i <text-folder>        -t vocab.json -o <name>.tokens\n" +
        "  node train.js pretrain  -i pretrain_data.tokens -t vocab.json -o pretrained.weights\n" +
        "  node train.js finetune  -i finetune_data.tokens -t vocab.json -p pretrained.weights -o finetuned.weights"
    );
    process.exit(1);
}

function main() {
    const command = process.argv[2];
    const args = parseArgs(process.argv.slice(3));

    if (command === "tokenizer") {
        buildVocabulary(args.input, args.output);
    } else if (command === "encode") {
        encodeCorpus(args.input, args.tokenizer, args.output);
    } else if (command === "pretrain") {
        pretrain(args.input, args.tokenizer, args.output);
    } else if (command === "finetune") {
        finetune(args.input, args.tokenizer, args.pretrained, args.output);
    } else {
        usage();
    }
}

// Every .txt file under a folder, in sorted order so a run is reproducible.
function listTextFiles(root) {
    const files = [];
    const stack = [root];
    while (stack.length > 0) {
        const dir = stack.pop();
        const entries = readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const full = join(dir, entry.name);
            if (entry.isDirectory()) {
                stack.push(full);
            } else if (entry.name.endsWith(".txt")) {
                files.push(full);
            }
        }
    }
    files.sort();
    return files;
}

// The six tokens the model reserves. Every one collapses to a single id.
const SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>", "<think>", "</think>"];

function buildVocabulary(inputDir, outputPath) {
    const files = listTextFiles(inputDir);

    // Read every file in the folder into one training string.
    const chunks = [];
    for (const file of files) {
        chunks.push(readFileSync(file, "utf-8"));
    }
    const text = chunks.join("");

    // Reserve the special tokens first, so they stay atomic during training,
    // then learn merges up to the target vocabulary size.
    const tokenizer = new Tokenizer();
    for (const token of SPECIAL_TOKENS) {
        tokenizer.reserve(token);
    }
    tokenizer.train(text, MODEL_VOCAB_SIZE);

    writeFileSync(outputPath, tokenizer.serializeToJSON());
    console.log(`vocab size ${tokenizer.vocabSize()}, wrote ${outputPath}`);
}

// Read a file's raw bytes into a fresh, aligned ArrayBuffer. readFileSync hands
// back a Buffer that may sit at an offset inside a shared pool, which a typed
// array cannot always view directly, so the bytes are copied out.
function readArrayBuffer(path) {
    const bytes = readFileSync(path);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

// Token ids are stored two bytes each, in the machine's native byte order, so a
// .tokens file is meant to be produced and read back on the same kind of machine.
function loadTokens(path) {
    return new Uint16Array(readArrayBuffer(path));
}

function loadTokenizer(path) {
    const tokenizer = new Tokenizer();
    tokenizer.deserializeFromJSON(readFileSync(path, "utf-8"));
    return tokenizer;
}

function encodeCorpus(inputDir, tokenizerPath, outputPath) {
    const tokenizer = loadTokenizer(tokenizerPath);
    const endOfText = tokenizer.encode("<|endoftext|>")[0];
    const files = listTextFiles(inputDir);

    // Encode one file at a time and append its ids to the file, followed by an
    // <|endoftext|> to mark the document boundary. Writing as we go keeps only
    // one document in memory at once, whatever the size of the corpus.
    const fd = openSync(outputPath, "w");
    let totalTokens = 0;
    try {
        for (let i = 0; i < files.length; ++i) {
            const ids = tokenizer.encode(readFileSync(files[i], "utf-8"));
            ids.push(endOfText);

            const packed = new Uint16Array(ids);
            writeSync(fd, Buffer.from(packed.buffer));
            totalTokens += ids.length;

            if ((i + 1) % 1000 === 0) {
                console.log(`encoded ${i + 1}/${files.length} files, ${totalTokens} tokens`);
            }
        }
    } finally {
        closeSync(fd);
    }

    console.log(`wrote ${totalTokens} tokens to ${outputPath}`);
}

function runTraining(trainer, examples, masks = null) {
    const LOG_EVERY = 10;
    let totalLoss = 0.0;

    if (masks !== null && masks.length !== examples.length) {
        throw new Error(
            `mask count ${masks.length} does not match ` +
            `example count ${examples.length}`
        );
    }

    for (let epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        for (let start = 0; start < examples.length; start += BATCH_SIZE) {
            const tokenIdBatch = examples.slice(start, start + BATCH_SIZE);
            const tokenMaskBatch =
                masks === null ? null : masks.slice(start, start + BATCH_SIZE);

            const loss = trainer.train(tokenIdBatch, tokenMaskBatch);
            totalLoss += loss;

            if (trainer.step % LOG_EVERY === 0) {
                const average = totalLoss / trainer.step;
                console.log(
                    `step ${trainer.step}/${trainer.totalSteps}, ` +
                    `lr ${trainer.lastLearningRate.toFixed(6)}, ` +
                    `loss ${loss.toFixed(4)}, avg ${average.toFixed(4)}`
                );
            }
        }
    }
}

function pretrain(tokensPath, tokenizerPath, outputPath) {
    const tokenizer = loadTokenizer(tokenizerPath);
    if (tokenizer.vocabSize() > MODEL_VOCAB_SIZE) {
        throw new Error(`tokenizer vocab ${tokenizer.vocabSize()} exceeds model vocab ${MODEL_VOCAB_SIZE}`);
    }

    const tokens = loadTokens(tokensPath);
    const model = buildModel();

    // Slice the stream into non-overlapping windows of CONTEXT_LENGTH + 1. The
    // extra token gives the last input in a window a next-token target. The
    // windows are views, so this does not copy the corpus.
    const windowLength = CONTEXT_LENGTH + 1;
    const windows = [];
    for (let start = 0; start + 1 < tokens.length; start += CONTEXT_LENGTH) {
        const window = tokens.subarray(start, start + windowLength);
        if (window.length >= 2) {
            windows.push(window);
        }
    }

    if (windows.length === 0) {
        throw new Error("No pretraining windows were produced");
    }

    const totalSteps =
        NUM_EPOCHS * Math.ceil(windows.length / BATCH_SIZE);
    const trainer = new AdamWTrainer(model, {
        maxLearningRate: 3e-4,
        minLearningRate: 3e-5,
        warmupSteps: Math.floor(totalSteps * 0.02),
        totalSteps,
        weightDecay: 0.01,
        gradientClip: 1.0,
    });

    console.log(
        `pretraining on ${windows.length} windows of up to ${CONTEXT_LENGTH} tokens, ` +
        `batch size ${BATCH_SIZE}, ${totalSteps} steps`
    );
    runTraining(trainer, windows); // no masks: every token is a target

    writeFileSync(outputPath, Buffer.from(model.serializeToArrayBuffer()));
    console.log(`wrote weights to ${outputPath}`);
}

// A mask that is 1 where a token belongs to an assistant reply - its text and
// the <|end|> that closes it - and 0 everywhere else. Those are the only tokens
// the model is graded on, so it learns to produce the assistant's side of the
// conversation rather than to echo back the user's.
function buildLossMask(tokens, special) {
    const mask = new Uint8Array(tokens.length);
    let inAssistant = false;

    for (let i = 0; i < tokens.length; ++i) {
        const id = tokens[i];

        if (id === special.assistant) {
            inAssistant = true; // the reply starts after this marker
            mask[i] = 0;        // but producing the marker itself is not graded
        } else if (id === special.user || id === special.endOfText) {
            inAssistant = false;
            mask[i] = 0;
        } else if (id === special.end) {
            mask[i] = inAssistant ? 1 : 0; // grade the reply's closing <|end|>
            inAssistant = false;
        } else {
            mask[i] = inAssistant ? 1 : 0;
        }
    }

    return mask;
}

// Split the stream into documents on the <|endoftext|> separators. The pieces
// are views, so no token data is copied.
function splitDocuments(tokens, endOfText) {
    const documents = [];
    let start = 0;
    for (let i = 0; i < tokens.length; ++i) {
        if (tokens[i] === endOfText) {
            if (i > start) {
                documents.push(tokens.subarray(start, i));
            }
            start = i + 1;
        }
    }
    if (start < tokens.length) {
        documents.push(tokens.subarray(start, tokens.length));
    }
    return documents;
}

// A document has something to teach only if at least one of its target
// positions is graded. The first token is never a target, so the scan starts
// at index one, matching how the trainer shifts the mask.
function hasGradedTarget(mask) {
    for (let i = 1; i < mask.length; ++i) {
        if (mask[i]) {
            return true;
        }
    }
    return false;
}

function finetune(tokensPath, tokenizerPath, pretrainedPath, outputPath) {
    const tokenizer = loadTokenizer(tokenizerPath);
    const special = {
        user: tokenizer.encode("<|user|>")[0],
        assistant: tokenizer.encode("<|assistant|>")[0],
        end: tokenizer.encode("<|end|>")[0],
        endOfText: tokenizer.encode("<|endoftext|>")[0],
    };

    const tokens = loadTokens(tokensPath);
    const documents = splitDocuments(tokens, special.endOfText);

    // Build one training example and mask per usable document. A document with
    // no graded target is dropped: it would contribute nothing and divide by zero.
    const examples = [];
    const masks = [];
    for (let i = 0; i < documents.length; ++i) {
        const document = documents[i];
        if (document.length < 2) {
            continue;
        }
        const mask = buildLossMask(document, special);
        if (!hasGradedTarget(mask)) {
            continue;
        }
        examples.push(document);
        masks.push(mask);
    }

    if (examples.length === 0) {
        throw new Error("No usable fine-tuning chats were found");
    }

    // Start from the pretrained weights.
    const model = buildModel();
    model.deserializeFromArrayBuffer(readArrayBuffer(pretrainedPath));

    // A gentler rate than pretraining, so fine-tuning nudges the model toward
    // the chat format without washing out what pretraining taught it.
    const totalSteps =
        NUM_EPOCHS * Math.ceil(examples.length / BATCH_SIZE);
    const trainer = new AdamWTrainer(model, {
        maxLearningRate: 5e-5,
        minLearningRate: 5e-6,
        warmupSteps: Math.floor(totalSteps * 0.02),
        totalSteps,
        weightDecay: 0.01,
        gradientClip: 1.0,
    });

    console.log(
        `fine-tuning on ${examples.length} chats, ` +
        `batch size ${BATCH_SIZE}, ${totalSteps} steps`
    );
    runTraining(trainer, examples, masks);

    writeFileSync(outputPath, Buffer.from(model.serializeToArrayBuffer()));
    console.log(`wrote weights to ${outputPath}`);
}

main();
