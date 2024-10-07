# Testing stella numerical accuracy against pytorch implementation

## Prerequisites
Usual setup to run `transformers` (pytorch, safetensors, tokenizers ..)


## Steps:
- Download the `stella_en_1.5B_v5` data, skip downloading `pytorch_model.bin`
```
huggingface-cli download dunzhang/stella_en_1.5B_v5 --exclude "pytorch_model.bin" --local-dir stella-cache
```

- run
```
cd src/stella_num
./stella.sh
```

## Implementation:
- For a set of `texts` the shell script will first run the `pytorch` based model and save the results as `stella-<embed dim>.safetensors`
- Then the script will call `cargo test stella_run --release -- --nocapture`, this will run a test that will generate `candle` outputs and compare results
- During comparison both `torch` and `candle` generated tensors are adjusted for `tolerence`. # TODO: reverse this and check with `torch.all_close` instead
```
let tolerence = 100.;
let t = ((t * tolerence).unwrap().round().unwrap() / tolerence).unwrap();
```

## Results:
DType::F32, the results are accurate to upto a tolerence of 3 decimal places. Diverges beyond that.