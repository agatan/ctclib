use std::fs::File;

use ctclib::{Dict, KenLM, LM};

#[test]
fn kenlm_model_works() {
    let dict = Dict::parse(File::open("data/letter.dict").unwrap()).unwrap();
    let mut kenlm = KenLM::new("data/overfit.arpa", &dict);
    let root = kenlm.start();
    let score = kenlm.score(&root, dict.index("M").unwrap());
    assert_eq!(score, -0.045306083);
    let next_state = kenlm.next_state(&root, dict.index("M").unwrap());
    {
        let score = kenlm.finish(&next_state);
        assert_eq!(score, -2.9529781);
    }
    {
        let next_state = kenlm.next_state(&next_state, dict.index("M").unwrap());
        let score = kenlm.finish(&next_state);
        assert_eq!(score, -1.6666714);
    }
}
