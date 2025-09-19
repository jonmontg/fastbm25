use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;

#[pyclass]
struct BM25 {
    corpus_size: usize,
    avgdl: f64, // average tokens per document
    doc_freqs: Vec<HashMap<u32, u32>>, // frequency of words in each document
    idf: HashMap<u32, f64>,
    doc_len: Vec<u32>,
    k1: f64,
    b: f64,
}

impl BM25 {
    fn scores_for_query(&self, query: &[u32]) -> Vec<f64> {
        let mut scores = vec![0.0; self.corpus_size];
        if self.corpus_size == 0 || self.avgdl == 0.0 {
            return scores;
        }
        let denom_base: Vec<f64> = self.doc_len.iter()
            .map(|&dl| self.k1 * (1.0 - self.b + self.b * (dl as f64) / self.avgdl))
            .collect();

        for q in query {
            let idf = *self.idf.get(q).unwrap_or(&0.0);
            if idf == 0.0 { continue; }
            for (i, tf_map) in self.doc_freqs.iter().enumerate() {
                if let Some(&tf) = tf_map.get(q) {
                    let tf = tf as f64;
                    let denom = tf + denom_base[i];
                    if denom > 0.0 {
                        scores[i] += idf * (tf * (self.k1 + 1.0) / denom);
                    }
                }
            }
        }
        scores
    }
}

#[pymethods]
impl BM25 {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75))]
    fn new(corpus: Vec<Vec<u32>>, k1: f64, b: f64) -> Self {
        let corpus_size = corpus.len();

        let mut nd: HashMap<u32, u32> = HashMap::new(); // word -> num docs with word
        let mut doc_freqs: Vec<HashMap<u32, u32>> = Vec::with_capacity(corpus.len());
        let mut doc_len: Vec<u32> = Vec::with_capacity(corpus.len());

        // Build the metadata
        for document in &corpus {
            let mut word_freqs: HashMap<u32, u32> = HashMap::new();
            for &word in document {
                *word_freqs.entry(word).or_insert(0) += 1;
            }

            doc_len.push(document.len() as u32);

            for &word in word_freqs.keys() {
                *nd.entry(word).or_insert(0) += 1;
            }
            doc_freqs.push(word_freqs);
        }

        let total_tokens: u64 = doc_len.iter().map(|&x| x as u64).sum();
        let avgdl = if corpus_size > 0 {
            (total_tokens as f64) / (corpus_size as f64)
        } else {
            0.0
        };

        // Build the IDF index
        let mut idf: HashMap<u32, f64> = HashMap::with_capacity(nd.len());
        let mut idf_sum: f64 = 0.0;
        let mut negative_idfs: Vec<u32> = Vec::new();
        let n = corpus_size as f64;

        for (word, &df_u32) in &nd {
            let df = df_u32 as f64;

            let widf = (n - df + 0.5).ln() - (df + 0.5).ln();
            idf_sum += widf;
            if widf < 0.0 {
                negative_idfs.push(*word);
            }
            idf.insert(*word, widf);
        }

        let average_idf = if !idf.is_empty() {
            idf_sum / (idf.len() as f64)
        } else {
            0.0
        };

        let eps = 0.25 * average_idf;
        for word in negative_idfs {
            idf.insert(word, eps);
        }

        // Done!
        Self {
            corpus_size,    // corpus_size
            avgdl,          // avgdl
            doc_freqs,      // doc_freqs
            idf,            // idf
            doc_len,        // doc_len
            k1,
            b,
        }
    }

    #[pyo3(signature = (query))]
    fn get_scores(&self, query: Vec<u32>) -> Vec<f64> {
        self.scores_for_query(&query)
    }

    #[pyo3(signature = (query, k))]
    fn get_top_k_indices(&self, query: Vec<u32>, k: usize) -> Vec<usize> {
        let scores = self.get_scores(query);
        let n = scores.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);

        // Treat NaN as -inf so they never become "top"
        let key = |x: f64| if x.is_nan() { f64::NEG_INFINITY } else { x };

        // If the highest score is exactly 0.0, return the sentinel indices
        let max_score = scores.iter().copied().map(key).fold(f64::NEG_INFINITY, f64::max);
        if max_score == 0.0 {
            return vec![n; k];  // sentinel: n == self.corpus_size
        }
        // (Alternative, a bit more robust: if max_score <= 0.0 && max_score.is_finite() { ... })

        // Normal top-k on indices
        let nth = n - k;
        let mut idx: Vec<usize> = (0..n).collect();

        idx.select_nth_unstable_by(nth, |&a, &b| key(scores[a]).total_cmp(&key(scores[b])));

        let mut topk = idx.split_off(nth);
        topk.sort_by(|&a, &b| key(scores[b]).total_cmp(&key(scores[a])));
        topk
    }
}

#[pymodule]
fn fastbm25(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}
