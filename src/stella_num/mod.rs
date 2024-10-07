use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stella_en_v5::{Config, EmbedDim, EmbeddingModel};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::select_device;

fn generate(
    device: &Device,
    model: &mut EmbeddingModel,
    tokenizer: &Tokenizer,
    text: &[&str],
) -> Tensor {
    let mut encoded = tokenizer.encode_batch(text.to_vec(), true).unwrap();

    // Now, we generate the tensors for the `input` and `mask`
    let shape = (encoded.len(), encoded[1].len());
    let mut ids = Tensor::zeros(shape, DType::U32, device).unwrap();
    let mut masks = Tensor::zeros(shape, DType::U8, device).unwrap();

    for (i, e) in encoded.drain(..).enumerate() {
        let input_id = Tensor::from_iter(e.get_ids().to_vec(), device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mask = Tensor::from_iter(e.get_attention_mask().to_vec(), device)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        ids = ids
            .slice_assign(&[i..i + 1, 0..input_id.dims2().unwrap().1], &input_id)
            .unwrap();
        masks = masks
            .slice_assign(&[i..i + 1, 0..mask.dims2().unwrap().1], &mask)
            .unwrap();
    }

    model.forward_norm(&ids, &masks).unwrap()
}

fn stella_variant(e: EmbedDim, tokenizer: &Tokenizer, txt: Vec<&str>) {
    println!("---------------------------------------------------------");
    let device = select_device().unwrap();
    let base_path = Path::new(
        format!(
            "{}/src/stella_num/stella-cache/",
            env!("CARGO_MANIFEST_DIR")
        )
        .as_str(),
    )
    .to_path_buf();
    let base_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[base_path.join("model.safetensors")],
            candle_core::DType::F32,
            &device,
        )
        .unwrap()
    };

    let (embd_dim_w, pt_file) = {
        match e {
            EmbedDim::Dim256 => ("2_Dense_256", "stella_256.safetensors"),
            EmbedDim::Dim1024 => ("2_Dense_1024", "stella_1024.safetensors"),
            EmbedDim::Dim4096 => ("2_Dense_4096", "stella_4096.safetensors"),
            _ => unimplemented!(),
        }
    };

    let cfg = Config::new_1_5_b_v5(e);

    let embed_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[base_path.join(embd_dim_w).join("model.safetensors")],
            candle_core::DType::F32,
            &device,
        )
        .unwrap()
    };

    let mut model = EmbeddingModel::new(&cfg, base_vb, embed_vb).unwrap();
    let t = generate(&device, &mut model, tokenizer, &txt);
    let pt = candle_core::safetensors::load(
        format!("{}/src/stella_num/{pt_file}", env!("CARGO_MANIFEST_DIR")),
        &device,
    )
    .unwrap();

    let tolerence = 100.;

    let pt = ((pt.get("w").unwrap() * tolerence).unwrap().round().unwrap() / tolerence).unwrap();
    let t = ((t * tolerence).unwrap().round().unwrap() / tolerence).unwrap();

    assert_eq!(
        t.eq(&pt)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u8>()
            .unwrap(),
        0
    );
}

pub fn stella() {
    let tkfile = format!(
        "{}/src/stella_num/stella-cache/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    println!("{tkfile}");
    let mut tokenizer = Tokenizer::from_file(&tkfile).unwrap();
    let pad_id = if let Some(pad_id) = tokenizer.token_to_id("<|endoftext|>") {
        pad_id
    } else {
        panic!("Tokenizer doesn't contain expected `<|endoftext|>` token");
    };

    // This part is super important, we are padding the tokens to the *`left`* and not the usual *`right`* padding
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_id,
        pad_token: "<|endoftext|>".to_string(),
        ..Default::default()
    }));

    let datapath = format!("{}/src/stella_num/samples.txt", env!("CARGO_MANIFEST_DIR"));
    let txt = std::fs::read_to_string(&datapath).unwrap();
    let txt = txt.lines().collect::<Vec<_>>();

    let variants = [EmbedDim::Dim256, EmbedDim::Dim1024, EmbedDim::Dim4096];
    for v in variants.iter() {
        stella_variant(*v, &tokenizer, txt.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::stella;

    #[test]
    fn stella_run() {
        stella();
    }
}
