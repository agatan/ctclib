#define KENLM_MAX_ORDER 6

#include "kenlm/lm/word_index.hh"
#include "kenlm/lm/return.hh"
#include "kenlm/lm/state.hh"
#include "kenlm/lm/virtual_interface.hh"
#include "kenlm/util/mmap.hh"
#include "kenlm/lm/config.hh"
#include "kenlm/lm/model.hh"

extern "C" {
	lm::base::Model *lm_ngram_LoadVirtual(const char *filename, const lm::ngram::Config *config);

	lm::base::Model *lm_ngram_LoadVirtualWithDefaultConfig(const char *filename);
}
