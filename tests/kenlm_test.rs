use ctclib::{Decoder, DecoderOptions, KenLM, LM};

#[test]
fn kenlm_model_works() {
    let mut kenlm = KenLM::new("data/lm.arpa");
    let root = kenlm.start();
    let (next_state, score) = kenlm.score(&root, 2);
    assert_eq!(score, -99.0);
    let (_, score) = kenlm.finish(&next_state);
    assert_eq!(score, -2.348754);
}

#[test]
fn decoder_kenlm_integration_works() {
    let mut decoder = Decoder::new(
        DecoderOptions {
            beam_size: 200,
            beam_size_token: 200,
            beam_threshold: f32::MAX,
            lm_weight: 0.5,
        },
        1,
        KenLM::new("data/lm.arpa"),
    );
    decoder.decode()
}
