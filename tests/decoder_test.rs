use ctclib::{Decoder, DecoderOptions, ZeroLM};

#[test]
fn it_decodes_ctc_sequence_correctly() {
    let vocab = ["'", " ", "a", "b", "c", "d", "_"];
    #[rustfmt::skip]
    let seq1 = [
        0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.03612540, 0.18184413, 0.16493624,
        0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462, 0.00948930, 0.06890021,
        0.21810400, 0.19992557, 0.18245131, 0.08503348, 0.14903535, 0.08424043, 0.08120984,
        0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305, 0.05206269, 0.09772094,
        0.13333870, 0.00550838, 0.00301669, 0.21745861, 0.20803985, 0.41317442, 0.01946335,
        0.16468227, 0.19806990, 0.19065450, 0.18963251, 0.19860937, 0.04377724, 0.01457421,
    ].into_iter().map(|x: f32| x.ln()).collect::<Vec<f32>>();
    let blank = vocab.iter().position(|&r| r == "_").unwrap() as i32;

    let mut decoder = Decoder::new(
        DecoderOptions {
            beam_size: 1,
            beam_size_token: 2000000,
            beam_threshold: f32::MAX,
            lm_weight: 0.0,
        },
        blank,
        ZeroLM,
    );
    let outputs = decoder.decode(&seq1, seq1.len() / vocab.len(), vocab.len());
    let tokens = outputs[0]
        .reduced_tokens(blank)
        .into_iter()
        .map(|i| vocab[i as usize])
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(tokens, "ac'bdc");

    let mut decoder = Decoder::new(
        DecoderOptions {
            beam_size: 200,
            beam_size_token: 2000000,
            beam_threshold: f32::MAX,
            lm_weight: 0.0,
        },
        blank,
        ZeroLM,
    );
    let outputs = decoder.decode(&seq1, seq1.len() / vocab.len(), vocab.len());
    let tokens = outputs[0]
        .reduced_tokens(blank)
        .into_iter()
        .map(|i| vocab[i as usize])
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(tokens, "acb");
}
