use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use std::collections::HashMap;

/// BM25 (Best Matching 25) ranking algorithm implementation.
///
/// BM25 is a probabilistic ranking function used by search engines to estimate
/// the relevance of documents to a given search query. It's an improvement over
/// the TF-IDF (Term Frequency-Inverse Document Frequency) approach.
///
/// The algorithm considers:
/// - Term frequency (TF): How often a term appears in a document
/// - Inverse document frequency (IDF): How rare a term is across the corpus
/// - Document length normalization: Prevents longer documents from being unfairly favored
///
/// # Fields
///
/// * `corpus_size` - Total number of documents in the corpus
/// * `avgdl` - Average document length (in tokens) across the corpus
/// * `doc_freqs` - Term frequencies for each document (doc_id -> term_id -> frequency)
/// * `idf` - Inverse document frequency scores for each term (term_id -> idf_score)
/// * `doc_len` - Length of each document in tokens
/// * `k1` - Term frequency saturation parameter (controls how quickly TF saturates)
/// * `b` - Length normalization parameter (controls how much document length affects scoring)
#[pyclass]
struct BM25 {
    /// Total number of documents in the corpus
    corpus_size: usize,
    /// Average document length (in tokens) across the corpus
    avgdl: f64,
    /// Term frequencies for each document (doc_id -> term_id -> frequency)
    doc_freqs: Vec<HashMap<u32, u32>>,
    /// Inverse document frequency scores for each term (term_id -> idf_score)
    idf: HashMap<u32, f64>,
    /// Length of each document in tokens
    doc_len: Vec<u32>,
    /// Term frequency saturation parameter (default: 1.5)
    k1: f64,
    /// Length normalization parameter (default: 0.75)
    b: f64,
}

#[pymethods]
impl BM25 {
    /// Creates a new BM25 instance from a corpus of documents.
    ///
    /// # Arguments
    ///
    /// * `corpus` - A vector of documents, where each document is a vector of term IDs (u32)
    /// * `k1` - Term frequency saturation parameter (default: 1.5)
    ///   - Controls how quickly term frequency saturates
    ///   - Higher values mean TF has more impact on scoring
    ///   - Typical range: 1.2 to 2.0
    /// * `b` - Length normalization parameter (default: 0.75)
    ///   - Controls how much document length affects scoring
    ///   - 0.0 = no length normalization, 1.0 = full normalization
    ///   - Typical range: 0.0 to 1.0
    ///
    /// # Returns
    ///
    /// A new BM25 instance with precomputed IDF scores and document statistics.
    ///
    /// # Panics
    ///
    /// This function will not panic, but empty corpora will result in a BM25 instance
    /// that returns zero scores for all queries.
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75))]
    fn new(corpus: Vec<Vec<u32>>, k1: f64, b: f64) -> Self {
        let corpus_size = corpus.len();

        // Parallel processing of documents to build term frequencies and lengths
        // Use enumerate to preserve document order - critical for maintaining
        // correct document indices that map back to the original corpus
        let document_data: Vec<(usize, HashMap<u32, u32>, u32, HashMap<u32, u32>)> = corpus
            .par_iter()
            .enumerate()
            .map(|(doc_idx, document)| {
                // Count term frequencies within this document
                let mut word_freqs: HashMap<u32, u32> = HashMap::new();
                for &word in document {
                    *word_freqs.entry(word).or_insert(0) += 1;
                }

                // Create a map of unique terms in this document for document frequency counting
                let mut doc_terms: HashMap<u32, u32> = HashMap::new();
                for &word in document {
                    doc_terms.insert(word, 1); // Just mark presence, not frequency
                }

                (doc_idx, word_freqs, document.len() as u32, doc_terms)
            })
            .collect();

        // Extract the three components in correct order
        let mut doc_freqs: Vec<HashMap<u32, u32>> = Vec::with_capacity(document_data.len());
        let mut doc_len: Vec<u32> = Vec::with_capacity(document_data.len());
        let mut document_frequencies: Vec<HashMap<u32, u32>> =
            Vec::with_capacity(document_data.len());

        for (_, word_freqs, length, doc_terms) in document_data {
            doc_freqs.push(word_freqs);
            doc_len.push(length);
            document_frequencies.push(doc_terms);
        }

        // Aggregate document frequencies across all documents
        let mut nd: HashMap<u32, u32> = HashMap::new();
        for doc_terms in document_frequencies {
            for (word, _) in doc_terms {
                *nd.entry(word).or_insert(0) += 1;
            }
        }

        // Calculate average document length for length normalization
        let total_tokens: u64 = doc_len.iter().map(|&x| x as u64).sum();
        let avgdl = if corpus_size > 0 {
            (total_tokens as f64) / (corpus_size as f64)
        } else {
            0.0
        };

        // Parallel computation of IDF scores
        let n = corpus_size as f64;
        let idf_entries: Vec<(u32, f64)> = nd
            .par_iter()
            .map(|(word, &df_u32)| {
                let df = df_u32 as f64; // Document frequency (number of docs containing this term)

                // IDF formula: log((N - df + 0.5) / (df + 0.5))
                // where N = total documents, df = documents containing the term
                let widf = (n - df + 0.5).ln() - (df + 0.5).ln();
                (*word, widf)
            })
            .collect();

        // Build IDF map and collect statistics
        let mut idf: HashMap<u32, f64> = HashMap::with_capacity(idf_entries.len());
        let mut idf_sum: f64 = 0.0;
        let mut negative_idfs: Vec<u32> = Vec::new();

        for (word, widf) in idf_entries {
            idf_sum += widf;

            // Track terms with negative IDF (appear in more than half the documents)
            if widf < 0.0 {
                negative_idfs.push(word);
            }
            idf.insert(word, widf);
        }

        // Handle negative IDFs by replacing them with a small positive value
        // This prevents terms that appear in many documents from having negative impact
        let average_idf = if !idf.is_empty() {
            idf_sum / (idf.len() as f64)
        } else {
            0.0
        };

        // Replace negative IDFs with 25% of the average IDF
        let eps = 0.25 * average_idf;
        for word in negative_idfs {
            idf.insert(word, eps);
        }

        // Done!
        Self {
            corpus_size,
            avgdl,
            doc_freqs,
            idf,
            doc_len,
            k1,
            b,
        }
    }

    /// Calculates BM25 relevance scores for a query against all documents in the corpus.
    ///
    /// This method uses parallel processing to efficiently calculate scores across all
    /// documents in the corpus, providing significant performance improvements for
    /// large corpora and multi-core systems.
    ///
    /// # Arguments
    ///
    /// * `query` - A vector of term IDs representing the search query
    ///
    /// # Returns
    ///
    /// A vector of BM25 scores, one for each document in the corpus. Higher scores
    /// indicate greater relevance to the query. Scores are non-negative and can be zero
    /// for documents that don't contain any query terms.
    ///
    /// # Algorithm
    ///
    /// For each document, the BM25 score is calculated as:
    /// ```
    /// score = Σ IDF(term) * (TF(term) * (k1 + 1)) / (TF(term) + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    /// ```
    ///
    /// Where:
    /// - IDF(term) = Inverse Document Frequency of the term
    /// - TF(term) = Term Frequency in the document
    /// - k1 = Term frequency saturation parameter
    /// - b = Length normalization parameter
    /// - doc_len = Document length in tokens
    /// - avg_doc_len = Average document length in the corpus
    ///
    /// # Performance
    ///
    /// The document processing is parallelized using Rayon, which provides:
    /// - Linear scaling with CPU cores for large corpora
    /// - Optimal performance for queries with multiple terms
    /// - Consistent results across different hardware configurations
    #[pyo3(signature = (query))]
    fn get_scores(&self, query: Vec<u32>) -> Vec<f64> {
        // Initialize scores vector with zeros
        let mut scores = vec![0.0; self.corpus_size];

        // Handle edge cases: empty corpus or zero average document length
        if self.corpus_size == 0 || self.avgdl == 0.0 {
            return scores;
        }

        // Pre-compute the length normalization denominator base for each document
        // This is the k1 * (1 - b + b * (doc_len / avg_doc_len)) part of the formula
        let denom_base: Vec<f64> = self
            .doc_len
            .iter()
            .map(|&dl| self.k1 * (1.0 - self.b + self.b * (dl as f64) / self.avgdl))
            .collect();

        // Filter query terms that exist in the corpus and get their IDF scores
        let query_terms: Vec<(u32, f64)> = query
            .into_iter()
            .filter_map(|q| {
                let idf = *self.idf.get(&q)?;
                if idf != 0.0 { Some((q, idf)) } else { None }
            })
            .collect();

        // Parallel processing: calculate BM25 scores for all documents in parallel
        scores = (0..self.corpus_size)
            .into_par_iter()
            .map(|doc_idx| {
                let mut doc_score = 0.0;

                // Calculate contribution of each query term to this document
                for (term_id, idf) in &query_terms {
                    if let Some(&tf) = self.doc_freqs[doc_idx].get(term_id) {
                        let tf = tf as f64;
                        let denom = tf + denom_base[doc_idx];

                        if denom > 0.0 {
                            doc_score += idf * (tf * (self.k1 + 1.0) / denom);
                        }
                    }
                }

                doc_score
            })
            .collect();

        scores
    }

    /// Returns the indices of the top-k most relevant documents for a query.
    ///
    /// This method calculates BM25 scores for all documents and returns the indices
    /// of the k documents with the highest scores, excluding documents with zero scores.
    ///
    /// # Arguments
    ///
    /// * `query` - A vector of term IDs representing the search query
    /// * `k` - The number of top documents to return
    ///
    /// # Returns
    ///
    /// A vector of document indices (0-based) sorted by relevance in descending order.
    /// Only documents with non-zero BM25 scores are included. If fewer than k documents
    /// have non-zero scores, returns all such documents. If no documents have non-zero
    /// scores, returns an empty vector.
    ///
    /// # Examples
    ///
    /// ```python
    /// # Get top 3 most relevant documents
    /// top_docs = bm25.get_top_k_indices(query, k=3)
    /// # Returns something like [1, 2, 0] for the 3 most relevant documents
    /// ```
    #[pyo3(signature = (query, k))]
    fn get_top_k_indices(&self, query: Vec<u32>, k: usize) -> Vec<usize> {
        // Calculate BM25 scores for all documents
        let scores = self.get_scores(query);
        let n = scores.len();

        // Handle edge cases: empty corpus or k=0
        if n == 0 || k == 0 {
            return Vec::new();
        }

        // Filter out documents with zero scores and create (index, score) pairs
        // This ensures we only return documents that actually match the query
        let mut non_zero_indices: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .filter(|(_, score)| **score > 0.0)
            .map(|(idx, score)| (idx, *score))
            .collect();

        // If no documents have non-zero scores, return empty result
        if non_zero_indices.is_empty() {
            return Vec::new();
        }

        // Adjust k to not exceed the number of non-zero documents
        let k = k.min(non_zero_indices.len());

        // Sort by score in descending order (highest scores first)
        non_zero_indices.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Extract and return the top k document indices
        non_zero_indices
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Python module initialization function.
///
/// This function is called when the Python module is imported. It registers
/// the BM25 class with the Python interpreter, making it available for use
/// in Python code.
///
/// # Arguments
///
/// * `m` - A reference to the Python module being initialized
///
/// # Returns
///
/// * `PyResult<()>` - Ok(()) on success, or a Python exception on failure
///
/// # Example
///
/// ```python
/// import fastbm25
/// bm25 = fastbm25.BM25(corpus)
/// ```
#[pymodule]
fn fastbm25(m: &Bound<PyModule>) -> PyResult<()> {
    // Register the BM25 class with Python
    m.add_class::<BM25>()?;
    Ok(())
}
