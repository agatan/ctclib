#define KENLM_MAX_ORDER 6

#include "kenlm/lm/word_index.hh"
#include "kenlm/lm/return.hh"
#include "kenlm/lm/state.hh"
#include "kenlm/lm/virtual_interface.hh"
#include "kenlm/util/mmap.hh"
#include "kenlm/lm/config.hh"
#include "kenlm/lm/model.hh"

// bindgen does not generate inline functions, so we need to define them here.
extern "C"
{
    lm::base::Model *lm_ngram_LoadVirtual(const char *filename, const lm::ngram::Config *config);
    lm::base::Model *lm_ngram_LoadVirtualWithDefaultConfig(const char *filename);

    lm::WordIndex lm_base_Vocabulary_BeginSentence(const lm::base::Vocabulary *vocab);
    lm::WordIndex lm_base_Vocabulary_EndSentence(const lm::base::Vocabulary *vocab);
    lm::WordIndex lm_base_Vocabulary_Index(const lm::base::Vocabulary *vocab, const char* str, size_t len);

    float lm_base_Model_BaseScore(lm::base::Model *model, const void *instate, lm::WordIndex new_word, void *outstate);
    const lm::base::Vocabulary *lm_base_Model_BaseVocabulary(lm::base::Model *model);
    void lm_base_Model_BeginSentenceWrite(lm::base::Model *model, void *outstate);
    void lm_base_Model_delete(lm::base::Model *model);
}
