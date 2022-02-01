#include "wrapper.h"

extern "C" {
	lm::base::Model *lm_ngram_LoadVirtual(const char *filename, const lm::ngram::Config *config) {
		return lm::ngram::LoadVirtual(filename, *config);
	}

	lm::base::Model *lm_ngram_LoadVirtualWithDefaultConfig(const char *filename) {
		return lm::ngram::LoadVirtual(filename);
	}
}

