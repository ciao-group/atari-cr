import torch
from torch import nn
import numpy as np

from atari_cr.atari_head.dataset import GazeDataset, Mode
from atari_cr.atari_head.gaze_predictor import ArgParser
from atari_cr.common.module_overrides import ViTEmbedder
from atari_cr.atari_head.translation_transformer import Seq2SeqTransformer, train_epoch, evaluate, greedy_decode, PAD

if __name__ == "__main__":
    args = ArgParser().parse_args()

    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")
    torch.set_printoptions(sci_mode=False)
    np.random.seed(42)
    torch.manual_seed(42)

    # Plus 2 to make space for start and end token
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 84 * 84 + 2
    EMB_SIZE = 32 # Original: 768
    PATCH_SIZE = 21 # Original: 16 or 32
    SRC_SEQ_LENGTH = 84 // PATCH_SIZE + 1
    src_embedder = ViTEmbedder(
        image_size=84,
        patch_size=PATCH_SIZE,
        hidden_dim=EMB_SIZE,
        mlp_dim=1024, # Original: 3072
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    # Override conv layer to work with 4 channels instead of 3
    src_embedder.conv_proj = nn.Conv2d(4, EMB_SIZE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE).to(DEVICE)

    model = Seq2SeqTransformer(
        num_encoder_layers=4, 
        num_decoder_layers=4, 
        emb_size=EMB_SIZE,
        nhead=2,
        src_vocab_size=None, 
        tgt_vocab_size=NUM_CLASSES, 
        dim_feedforward=1024,
        custom_src_embedder=src_embedder
    )
    print(f"Number of model parameters: {sum([x.numel() for x in model.parameters()])}")

    # Loss function and optimizer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Create dataset and data loader
    env_name = "ms_pacman"
    single_run = "52_RZ_2394668_Aug-10-14-52-42" if args.debug else ""
    model_name = single_run or "all_trials"
    dataset = GazeDataset.from_atari_head_files(
        root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=single_run, 
        load_saliency=args.load_saliency, mode=Mode.GAZE_CLASSES)
    train_loader, val_loader = dataset.split(batch_size=64)

    # Train loop
    from timeit import default_timer as timer
    NUM_EPOCHS = 1
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        print("Training one epoch...")
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        end_time = timer()
        print("Evaluating...")
        val_loss = evaluate(model, val_loader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    src_mask = (torch.zeros(SRC_SEQ_LENGTH, SRC_SEQ_LENGTH)).type(torch.bool)
    frame_stack = torch.ones([1, 4, 84, 84]) # Batch with one stack of 4 greyscale frames
    greedy_decode(model, frame_stack, src_mask, )

# positional_encoder = PositionalEncoding(emb_size, dropout=0.1)
# tgt_embedder = nn.Embedding(num_classes, emb_size) # [7058, 128]

# # Init tranformer source and target
# BOS = torch.Tensor([[0]]) # Beginning of sequence token
# EOS = torch.Tensor([[1]]) # End of sequence token
# max_seq_length = 10
# # The input sequence is given by the [CLS] token following by 16 embeddings for the image patches
# frame_stack = torch.ones([1, 4, 84, 84]) # Batch with one stack of 4 greyscale frames
# ys = BOS # Start token

# # memory.shape: torch.Size([11, 1, 512])
# # out.shape: torch.Size([1, 1, 512])
# # prob.shape: torch.Size([1, 10837])
# # next_word: 6

# embedded_src = src_embedder(frame_stack)[0] # -> [17,128]
# memory = model.encoder(embedded_src) # -> [17,128]
# for i in range(100):
#     embedded_ys = positional_encoder(tgt_embedder(ys.long())) # -> [1,1,128]
#     output = model.decoder(embedded_ys, memory)
#     output = torch.argmax(ys, dim=-2)
#     if torch.all(output[:,i] == EOS): break
#     print(output.item(), end=", ")
# print("")

# # TODO: Training code