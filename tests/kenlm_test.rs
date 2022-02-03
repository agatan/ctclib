use ctclib::{KenLM, LM};

#[test]
fn kenlm_model_works() {
    let mut kenlm = KenLM::new("data/overfit.arpa");
    let root = kenlm.start();
    let (next_state, score) = kenlm.score(&root, 4);
    assert_eq!(score, -1.2412617);
    {
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -1.5861495);
    }
    {
        let (next_state, _) = kenlm.score(&next_state, 5);
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -1.9427716);
    }
}
