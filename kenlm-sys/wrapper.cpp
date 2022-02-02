#include "wrapper.h"

extern "C"
{
    lm::base::Model *lm_ngram_LoadVirtual(const char *filename, const lm::ngram::Config *config)
    {
        return lm::ngram::LoadVirtual(filename, *config);
    }

    lm::base::Model *lm_ngram_LoadVirtualWithDefaultConfig(const char *filename)
    {
        return lm::ngram::LoadVirtual(filename);
    }

    lm::WordIndex lm_base_Vocabulary_EndSentence(const lm::base::Vocabulary *vocab)
    {
        return vocab->EndSentence();
    }

    float lm_base_Model_BaseScore(lm::base::Model *model, const void *instate, lm::WordIndex new_word, void *outstate)
    {
        return model->BaseScore(instate, new_word, outstate);
    }

    const lm::base::Vocabulary *lm_base_Model_BaseVocabulary(lm::base::Model *model)
    {
        return &model->BaseVocabulary();
    }

    void lm_base_Model_NullContextWrite(lm::base::Model *model, void *outstate)
    {
        model->NullContextWrite(outstate);
    }
}
