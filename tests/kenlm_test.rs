use std::fs::File;

use ctclib::{Dict, KenLM, LM};

#[test]
fn kenlm_model_works() {
    let dict = Dict::parse(File::open("data/letter.dict").unwrap()).unwrap();
    let mut kenlm = KenLM::new("data/overfit.arpa", &dict);
    let root = kenlm.start();
    let (next_state, score) = kenlm.score(&root, dict.index("M").unwrap());
    assert_eq!(score, -1.3728311);
    {
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -1.6666714);
    }
    {
        let (next_state, _) = kenlm.score(&next_state, dict.index("I").unwrap());
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -2.8997345);
    }
}
