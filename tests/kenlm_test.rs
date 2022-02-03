use ctclib::{KenLM, LM};

#[test]
fn kenlm_model_works() {
    let mut kenlm = KenLM::new("data/lm.arpa");
    let root = kenlm.start();
    let (next_state, score) = kenlm.score(&root, 2);
    assert_eq!(score, -99.0);
    let (_, score) = kenlm.finish(&next_state);
    assert_eq!(score, -2.348754);
}
