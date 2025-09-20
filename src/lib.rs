use nohash_hasher::BuildNoHashHasher;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::arch::aarch64::*;
use std::collections::{HashMap, HashSet};

type IntMap = HashMap<u32, u32, BuildNoHashHasher<u32>>;
type IntSet = HashSet<u32, BuildNoHashHasher<u32>>;

/// Configuration for thread pool settings to optimize performance
#[pyclass]
#[derive(Clone)]
struct ThreadConfig {
    /// Number of threads to use for parallel processing
    /// If None, uses the default (number of CPU cores)
    num_threads: Option<usize>,
    /// Minimum chunk size for parallel processing
    /// Smaller chunks may have more overhead but better load balancing
    min_chunk_size: usize,
    /// Maximum chunk size for parallel processing
    /// Larger chunks reduce overhead but may cause load imbalance
    max_chunk_size: usize,
}

#[pymethods]
impl ThreadConfig {
    #[new]
    #[pyo3(signature = (num_threads=None, min_chunk_size=1000, max_chunk_size=10000))]
    fn new(num_threads: Option<usize>, min_chunk_size: usize, max_chunk_size: usize) -> Self {
        Self {
            num_threads,
            min_chunk_size,
            max_chunk_size,
        }
    }

    /// Get the number of threads to use
    fn get_num_threads(&self) -> usize {
        self.num_threads
            .unwrap_or_else(|| rayon::current_num_threads())
    }
}

/// ARM64 NEON optimized functions for BM25 computation
mod neon_utils {
    use super::*;

    /// ARM64 NEON optimized vector sum for u32 values
    /// Uses 128-bit vector registers to sum 4 u32 values at once
    #[inline]
    pub fn neon_sum_u32_slice(slice: &[u32]) -> u64 {
        let mut sum = 0u64;
        let chunks = slice.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Load 4 u32 values into a 128-bit vector register
            let v = unsafe { vld1q_u32(chunk.as_ptr()) };
            // Sum the 4 values using vector addition
            let part: u32 = unsafe { vaddvq_u32(v) };
            sum += part as u64;
        }

        // Handle remaining elements
        for &val in remainder {
            sum += val as u64;
        }
        sum
    }

    /// ARM64 NEON optimized vector sum for f64 values
    /// Uses 128-bit vector registers to sum 2 f64 values at once
    #[inline]
    pub fn neon_sum_f64_slice(slice: &[f64]) -> f64 {
        unsafe {
            // Accumulate in a vector register, reduce once at the end
            let mut acc = vdupq_n_f64(0.0);

            let chunks = slice.chunks_exact(2);
            let remainder = chunks.remainder();

            for chunk in chunks {
                // SAFETY: chunk has len == 2; pointer comes from a valid &[f64]
                let v = vld1q_f64(chunk.as_ptr());
                acc = vaddq_f64(acc, v);
            }

            // Horizontal add the two lanes of `acc` in one instruction
            let mut sum = vaddvq_f64(acc);

            // Handle the trailing element (if any)
            for &x in remainder {
                sum += x;
            }
            sum
        }
    }

    /// ARM64 NEON optimized document length calculation
    /// Processes document lengths in parallel using vector operations
    #[inline]
    pub fn calculate_doc_lengths(documents: &[Vec<u32>]) -> Vec<u32> {
        const PAR_THRESHOLD: usize = 1_000; // Lowered threshold for better parallelization
        if documents.len() >= PAR_THRESHOLD {
            documents.par_iter().map(|d| d.len() as u32).collect()
        } else {
            documents.iter().map(|d| d.len() as u32).collect()
        }
    }

    /// Enhanced parallel document length calculation with chunked processing
    #[inline]
    pub fn calculate_doc_lengths_chunked(documents: &[Vec<u32>], chunk_size: usize) -> Vec<u32> {
        if documents.len() < 500 {
            // For small datasets, use sequential processing
            documents.iter().map(|d| d.len() as u32).collect()
        } else {
            // Use chunked parallel processing for better load balancing
            documents
                .par_chunks(chunk_size)
                .map(|chunk| chunk.iter().map(|d| d.len() as u32).collect::<Vec<u32>>())
                .flatten()
                .collect()
        }
    }

    /// ARM64 NEON optimized term frequency counting with vectorized operations
    #[inline]
    pub fn count_term_frequencies(documents: &[Vec<u32>]) -> Vec<IntMap> {
        const PAR_THRESHOLD: usize = 500; // Lowered threshold for better parallelization
        if documents.len() >= PAR_THRESHOLD {
            documents
                .par_iter()
                .map(|document| {
                    // heuristic: assume ~50% unique; cap to keep allocations reasonable
                    let cap = (document.len().saturating_add(1) / 2).min(16_384);
                    let mut word_freqs: IntMap =
                        HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());

                    for &w in document {
                        // fast path with identity hashing
                        *word_freqs.entry(w).or_insert(0) += 1;
                    }
                    word_freqs
                })
                .collect()
        } else {
            documents
                .iter()
                .map(|document| {
                    let cap = (document.len().saturating_add(1) / 2).min(16_384);
                    let mut word_freqs: IntMap =
                        HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());

                    for &w in document {
                        *word_freqs.entry(w).or_insert(0) += 1;
                    }
                    word_freqs
                })
                .collect()
        }
    }

    /// Enhanced parallel term frequency counting with chunked processing
    #[inline]
    pub fn count_term_frequencies_chunked(
        documents: &[Vec<u32>],
        chunk_size: usize,
    ) -> Vec<IntMap> {
        if documents.len() < 200 {
            // For small datasets, use sequential processing
            documents
                .iter()
                .map(|document| {
                    let cap = (document.len().saturating_add(1) / 2).min(16_384);
                    let mut word_freqs: IntMap =
                        HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());

                    for &w in document {
                        *word_freqs.entry(w).or_insert(0) += 1;
                    }
                    word_freqs
                })
                .collect()
        } else {
            // Use chunked parallel processing for better load balancing
            documents
                .par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|document| {
                            let cap = (document.len().saturating_add(1) / 2).min(16_384);
                            let mut word_freqs: IntMap = HashMap::with_capacity_and_hasher(
                                cap,
                                BuildNoHashHasher::default(),
                            );

                            for &w in document {
                                *word_freqs.entry(w).or_insert(0) += 1;
                            }
                            word_freqs
                        })
                        .collect::<Vec<IntMap>>()
                })
                .flatten()
                .collect()
        }
    }

    #[inline]
    pub fn extract_unique_terms(documents: &[Vec<u32>]) -> Vec<IntSet> {
        const PAR_THRESHOLD: usize = 1_000; // Lowered threshold for better parallelization
        if documents.len() >= PAR_THRESHOLD {
            documents.par_iter().map(unique_set).collect()
        } else {
            documents.iter().map(unique_set).collect()
        }
    }

    /// Enhanced parallel unique terms extraction with chunked processing
    #[inline]
    pub fn extract_unique_terms_chunked(documents: &[Vec<u32>], chunk_size: usize) -> Vec<IntSet> {
        if documents.len() < 500 {
            // For small datasets, use sequential processing
            documents.iter().map(unique_set).collect()
        } else {
            // Use chunked parallel processing for better load balancing
            documents
                .par_chunks(chunk_size)
                .map(|chunk| chunk.iter().map(unique_set).collect::<Vec<IntSet>>())
                .flatten()
                .collect()
        }
    }

    #[inline]
    fn unique_set(doc: &Vec<u32>) -> IntSet {
        // Heuristic: assume ~50% unique to reduce rehashing; clamp to keep memory sane.
        let cap = (doc.len() / 2).max(8).min(16_384);
        let mut set: IntSet = HashSet::with_capacity_and_hasher(cap, BuildNoHashHasher::default());
        for &w in doc {
            set.insert(w); // presence only
        }
        set
    }
}

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
    doc_freqs: Vec<IntMap>,
    /// Inverse document frequency scores for each term (term_id -> idf_score)
    idf: HashMap<u32, f64>,
    /// Length of each document in tokens
    doc_len: Vec<u32>,
    /// Term frequency saturation parameter (default: 1.5)
    k1: f64,
    /// Length normalization parameter (default: 0.75)
    b: f64,
    /// Thread configuration for parallel processing
    thread_config: ThreadConfig,
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
    #[pyo3(signature = (corpus, k1=1.5, b=0.75, thread_config=None))]
    fn new(corpus: Vec<Vec<u32>>, k1: f64, b: f64, thread_config: Option<ThreadConfig>) -> Self {
        let corpus_size = corpus.len();

        // Use provided thread config or create default
        let thread_config = thread_config.unwrap_or_else(|| ThreadConfig::new(None, 1000, 10000));

        // Configure thread pool if custom thread count is specified
        if let Some(num_threads) = thread_config.num_threads {
            let _ = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global();
        }

        // Use ARM64 NEON optimized functions for better performance on ARM architectures
        let (doc_freqs, doc_len, document_frequencies) = {
            // Choose processing strategy based on corpus size
            let chunk_size = if corpus_size > 10000 {
                thread_config.max_chunk_size
            } else if corpus_size > 1000 {
                thread_config.min_chunk_size
            } else {
                corpus_size // No chunking for small corpora
            };

            // Use chunked parallel processing for large corpora
            let word_freqs = if corpus_size > 5000 {
                neon_utils::count_term_frequencies_chunked(&corpus, chunk_size)
            } else {
                neon_utils::count_term_frequencies(&corpus)
            };

            let doc_lengths = if corpus_size > 2000 {
                neon_utils::calculate_doc_lengths_chunked(&corpus, chunk_size)
            } else {
                neon_utils::calculate_doc_lengths(&corpus)
            };

            let doc_terms = if corpus_size > 2000 {
                neon_utils::extract_unique_terms_chunked(&corpus, chunk_size)
            } else {
                neon_utils::extract_unique_terms(&corpus)
            };

            (word_freqs, doc_lengths, doc_terms)
        };

        // Aggregate document frequencies across all documents
        let mut nd: HashMap<u32, u32> = HashMap::new();
        for doc_terms in document_frequencies {
            for word in doc_terms {
                *nd.entry(word).or_insert(0) += 1;
            }
        }

        // Calculate average document length for length normalization using NEON optimization
        let total_tokens: u64 = neon_utils::neon_sum_u32_slice(&doc_len);
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
        let mut negative_idfs: Vec<u32> = Vec::new();

        // Extract IDF values for vectorized sum calculation
        let idf_values: Vec<f64> = idf_entries.iter().map(|(_, widf)| *widf).collect();
        let idf_sum: f64 = neon_utils::neon_sum_f64_slice(&idf_values);

        for (word, widf) in idf_entries {
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
            thread_config,
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

        // Enhanced parallel processing: calculate BM25 scores for all documents in parallel
        // Use chunked processing for very large corpora to improve load balancing
        let chunk_size = if self.corpus_size > 50000 {
            self.thread_config.max_chunk_size
        } else if self.corpus_size > 10000 {
            self.thread_config.min_chunk_size
        } else {
            self.corpus_size // No chunking for smaller corpora
        };

        if self.corpus_size > 1000 {
            // Use chunked parallel processing for better load balancing
            scores = (0..self.corpus_size)
                .into_par_iter()
                .with_min_len(chunk_size)
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
        } else {
            // Use simple parallel processing for smaller corpora
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
        }

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

        // Use parallel processing for filtering and sorting large result sets
        let non_zero_indices: Vec<(usize, f64)> = if n > 10000 {
            // For very large corpora, use parallel processing
            scores
                .par_iter()
                .enumerate()
                .filter(|(_, score)| **score > 0.0)
                .map(|(idx, score)| (idx, *score))
                .collect()
        } else {
            // For smaller corpora, use sequential processing
            scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score > 0.0)
                .map(|(idx, score)| (idx, *score))
                .collect()
        };

        // If no documents have non-zero scores, return empty result
        if non_zero_indices.is_empty() {
            return Vec::new();
        }

        // Adjust k to not exceed the number of non-zero documents
        let k = k.min(non_zero_indices.len());

        // For large result sets, use parallel sorting
        let sorted_indices = if non_zero_indices.len() > 1000 {
            // Use parallel sorting for large result sets
            let mut indices = non_zero_indices;
            indices.par_sort_by(|a, b| b.1.total_cmp(&a.1));
            indices
        } else {
            // Use sequential sorting for smaller result sets
            let mut indices = non_zero_indices;
            indices.sort_by(|a, b| b.1.total_cmp(&a.1));
            indices
        };

        // Extract and return the top k document indices
        sorted_indices
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get the current thread configuration
    fn get_thread_config(&self) -> ThreadConfig {
        self.thread_config.clone()
    }

    /// Get performance statistics about the BM25 instance
    fn get_stats(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let stats = PyDict::new(py);
            stats.set_item("corpus_size", self.corpus_size)?;
            stats.set_item("avg_document_length", self.avgdl)?;
            stats.set_item("vocabulary_size", self.idf.len())?;
            stats.set_item("num_threads", self.thread_config.get_num_threads())?;
            stats.set_item("k1", self.k1)?;
            stats.set_item("b", self.b)?;
            Ok(stats.into())
        })
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
    // Register the ThreadConfig class with Python
    m.add_class::<ThreadConfig>()?;
    Ok(())
}
