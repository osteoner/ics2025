import pandas as pd
import numpy as np
import psycopg2,re
import os
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
import skfuzzy as fuzz
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import json
import torch
torch.cuda.empty_cache()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def preprocess_text(text):
  """
  Performs advanced data preprocessing steps for text cleaning.

  Args:
      text (str): The text to be preprocessed.

  Returns:
      str: The cleaned text.
  """
  text = re.sub(r"\\\\", "\n", text)
  quote_pattern = r'\*\*\*QUOTE\*\*\*([\s\S]*?)\*\*\*QUOTE\*\*\*'
  link_pattern = r'\*\*\*LINK\*\*\*([\s\S]*?)\*\*\*LINK\*\*\*'
  img_pattern = r'\*\*\*IMG\*\*\*([\s\S]*?)\*\*\*IMG\*\*\*'
  citing_pattern = r'\*\*\*CITING\*\*\*([\s\S]*?)\*\*\*CITING\*\*\*'
  iframe_pattern = r'\*\*\*IFRAME\*\*\*([\s\S]*?)\*\*\*IFRAME\*\*\*'
  attachment_pattern = r'\*\*\*ATTACHMENT\*\*\*([\s\S]*?)\*\*\*ATTACHMENT\*\*\*'
  text = re.sub(quote_pattern, 'QUOTE', text)
  text = re.sub(link_pattern, 'LINK', text)
  text = re.sub(img_pattern, 'IMG', text)
  text = re.sub(citing_pattern, 'CITING', text)
  text = re.sub(iframe_pattern, 'IFRAME', text)
  text = re.sub(attachment_pattern, 'ATTACHMENT', text)


  
  url_pattern = r'http\S+'

# Replace URLs with 'HTTPURL'
  text= re.sub(url_pattern, 'HTTPURL',text,flags=re.DOTALL)



  # Remove leading/trailing whitespace
  text = text.strip()
  text = re.sub(r'\n', ' ', text)

  # Replace \\\\ with \n for proper newline handling
  

  # Lowercase for case-insensitive processing
  text = text.lower()

  # Remove punctuation (consider using a custom list for specific needs)
  punctuation = r"[^\w\s'-]"  # Includes common punctuation, apostrophes, hyphens
  text = re.sub(punctuation, ' ', text)
  emoji_pattern = r"[^a-z0-9\s']+"  # Matches non-alphanumeric, non-whitespace, non-quote, non-hyphen characters
  text = re.sub(emoji_pattern, ' ', text)

  # Handle named entities (consider preserving or anonymizing based on privacy)
  # ... (code for named entity recognition and replacement/anonymization)

  # Handle typos and spelling errors (consider using a spell checker or dictionary)
  # ... (code for typo correction)

  # Handle HTML tags and entities (consider removing or converting)
  text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
  text = re.sub(r"&.+?;", " ", text)  # Remove HTML entities

  return text

class UserSequenceAnalysis:
    def __init__(self, db_config: Dict[str, str], embedding_models: List[str] = None, embeddings_dir: str = "embeddings"):
        """
        Initialize the UserSequenceAnalysis class.
        
        Args:
            db_config: Database configuration parameters
            embedding_models: List of embedding models to evaluate
        """
        self.db_config = db_config
        self.conn = None
        self.embeddings_dir = embeddings_dir
        # Ensure embeddings directory exists
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        self.data = {
            'threads': None,
            'posts': None,
            'members': None,
            'interactions': None
        }
        
        # Define embedding models to evaluate
        self.embedding_models = embedding_models or [
            'tfidf',
            # 'Xenova/text-embedding-ada-002',
            'bert-base-uncased',
            'all-MiniLM-L6-v2',
            "google/flan-t5-large"
                            ]
        
        # Model instances
        self.embedders = {}
        # self.topic_model = None
        
        # Representation configurations
        self.representations = {
            'R': 'Representation'
        }
        
        # Clustering algorithms
        self.clustering_algorithms = {
            'hdbscan': lambda min_cluster_size=5: HDBSCAN(min_cluster_size=min_cluster_size)
        }
        
        # Results storage
        self.user_sequences = {}
        self.embeddings = {}
        self.cluster_results = {}
        self.evaluation_results = {}
        
    def connect_to_db(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            logger.info("Connecting to PostgreSQL database...")
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Database connection established successfully.")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def load_data(self) -> None:
        """Load data from PostgreSQL database."""
        if not self.conn:
            self.connect_to_db()
        
        try:
            logger.info("Loading data from database...")
            
            # Load members data
            # self.data['members'] = pd.read_csv('breachforums_members.csv')
            self.data['members'] = pd.read_sql(
                """
                SELECT username, reputation, total_posts 
                FROM members where id > 0 limit 20000
                """, 
                self.conn
            )
            
            # Load threads data
            # self.data['threads'] = pd.read_csv('breachforums_threads.csv')
            self.data['threads'] = pd.read_sql(
                """
                SELECT id, creator, name
                FROM threads
                """, 
                self.conn
            )
            self.data['threads']['name'] = self.data['threads']['name'].astype('str').apply(preprocess_text)
            # Load posts data
            # self.data['posts'] = pd.read_csv('breachforums_posts.csv') 
            self.data['posts'] = pd.read_sql(
                """
                SELECT id, thread_id, creator, content
                FROM posts where is_initial_post = '1' 
                """, 
                self.conn
            )
            self.data['posts']['content'] = self.data['posts']['content'].astype('str').apply(preprocess_text)
            
            # Generate interactions data (users who interact the most in threads)
            # This is a derived table based on posts
            # interactions = pd.read_csv('breachforums_interactions.csv')
            interactions_query = """
                SELECT threads.creator as thread_creator, posts.creator as post_creator, 
                       COUNT(*) as interaction_count
                FROM posts
                JOIN threads ON posts.thread_id = threads.id
                WHERE threads.creator != posts.creator
                GROUP BY threads.creator, posts.creator
                ORDER BY thread_creator, interaction_count DESC
            """
            interactions = pd.read_sql(interactions_query, self.conn)
            
            # For each thread creator, get top 10 users who interact with them
            self.data['interactions'] = interactions.groupby('thread_creator').apply(
                lambda x: x.nlargest(10, 'interaction_count')
            ).reset_index(drop=True)
            
            logger.info("Data loaded successfully.")
            logger.info(f"Loaded {len(self.data['members'])} members, {len(self.data['threads'])} threads, "
                      f"{len(self.data['posts'])} posts.")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def generate_user_sequences(self):
        """Generate sequences for each user based on their activity data."""
        logger.info("Generating user sequences...")
        user_sequences = {
            'R': {}
        }
        
        # Convert to dictionaries for faster lookup
        threads_df = self.data['threads']
        posts_df = self.data['posts']
        interactions_df = self.data['interactions']
        
        # Pre-calculate username groups to avoid repeated filtering
        threads_by_creator = dict(list(threads_df.groupby('creator')))
        posts_by_creator = dict(list(posts_df.groupby('creator')))
        interactions_by_thread_creator = dict(list(interactions_df.groupby('thread_creator')))
        
        # Get list of usernames once
        members_data = self.data['members'].to_dict('records')
        
        for member in tqdm(members_data, desc="Generating user sequences"):
            username = member['username']
            
            # Generate metadata once
            metadata = f"[M]{username}[SEP]{member['reputation']}[SEP]{member['total_posts']}"
            
            # Process user's threads - avoid filtering the dataframe repeatedly
            user_thread_df = threads_by_creator.get(username, pd.DataFrame())
            thread_texts = []
            if not user_thread_df.empty:
                thread_texts = [f"[T]{preprocess_text(text)}" for text in user_thread_df['name'].astype(str)[:10]]
            
            # Process user's posts (replies) - avoid filtering the dataframe repeatedly
            user_posts_df = posts_by_creator.get(username, pd.DataFrame())
            post_texts = []
            if not user_posts_df.empty:
                post_texts = [f"[R]{preprocess_text(text)}" for text in user_posts_df['content'].astype(str)[:15]]
            
            # Process user interactions - avoid filtering the dataframe repeatedly
            interaction_texts = []
            user_interactions_df = interactions_by_thread_creator.get(username, pd.DataFrame())
            if not user_interactions_df.empty:
                # Use nlargest directly on the dataframe instead of filtering first
                top_interactors = user_interactions_df.nlargest(10, 'interaction_count')
                interaction_texts = [
                    f"[I]{row['post_creator']}" 
                    for _, row in top_interactors.iterrows()
                ]
            
            # Combine all text elements
            user_sequences['R'][username] = [metadata] + thread_texts + post_texts + interaction_texts
        
        # Store the sequences
        self.user_sequences = user_sequences
        
        logger.info(f"Generated sequences for {len(user_sequences['R'])} users.")
        return user_sequences
    
    def _join_user_sequence(self, sequence: List[str]) -> str:
        """Join a user sequence into a single string for embedding."""
        return " ".join(sequence)

    def _get_embedding_filename(self, representation: str, model_name: str) -> str:
        """Generate filename for embedding file."""
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        return os.path.join(self.embeddings_dir, f"embeddings_{representation}_{safe_model_name}.npy")
    
    def _get_usernames_filename(self, representation: str, model_name: str) -> str:
        """Generate filename for usernames file (to maintain order)."""
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        return os.path.join(self.embeddings_dir, f"usernames_{representation}_{safe_model_name}.json")
    
    def _save_embeddings(self, embeddings_dict: Dict[str, np.ndarray], representation: str, model_name: str) -> None:
        """Save embeddings to disk."""
        try:
            usernames = list(embeddings_dict.keys())
            embeddings_matrix = np.array([embeddings_dict[username] for username in usernames])
            
            # Save embeddings matrix
            embedding_file = self._get_embedding_filename(representation, model_name)
            np.save(embedding_file, embeddings_matrix)
            
            # Save usernames order
            usernames_file = self._get_usernames_filename(representation, model_name)
            with open(usernames_file, 'w') as f:
                json.dump(usernames, f)
            
            logger.info(f"Saved embeddings for {representation}-{model_name} to {embedding_file}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings for {representation}-{model_name}: {e}")
    
    def _load_embeddings(self, representation: str, model_name: str) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from disk if they exist."""
        try:
            embedding_file = self._get_embedding_filename(representation, model_name)
            usernames_file = self._get_usernames_filename(representation, model_name)
            
            # Check if both files exist
            if not (os.path.exists(embedding_file) and os.path.exists(usernames_file)):
                return None
            
            # Load embeddings matrix
            embeddings_matrix = np.load(embedding_file)
            
            # Load usernames order
            with open(usernames_file, 'r') as f:
                usernames = json.load(f)
            
            # Convert back to dictionary
            embeddings_dict = {
                username: emb for username, emb in zip(usernames, embeddings_matrix)
            }
            
            logger.info(f"Loaded existing embeddings for {representation}-{model_name} from {embedding_file}")
            return embeddings_dict
            
        except Exception as e:
            logger.warning(f"Error loading embeddings for {representation}-{model_name}: {e}")
            return None
    
    def _embeddings_exist(self, representation: str, model_name: str) -> bool:
        """Check if embeddings exist for given representation and model."""
        embedding_file = self._get_embedding_filename(representation, model_name)
        usernames_file = self._get_usernames_filename(representation, model_name)
        return os.path.exists(embedding_file) and os.path.exists(usernames_file)
    
    def initialize_embedding_models(self) -> None:
        """Initialize embedding models and add special tokens to SentenceTransformer tokenizers."""
        logger.info("Initializing embedding models...")
        
        # Define special tokens
        special_tokens = ['[M]', '[T]', '[R]', '[I]', '[SEP]']
        
        for model_name in self.embedding_models:
            if model_name == 'tfidf':
                self.embedders[model_name] = TfidfVectorizer(max_features=300)
            else:
                try:
                    # Initialize SentenceTransformer with the actual model name
                    logger.info(f"Loading model: {model_name}")
                    model = SentenceTransformer(model_name)
                    
                    # Add special tokens to the tokenizer
                    tokenizer = model.tokenizer
                    num_added_tokens = tokenizer.add_tokens(special_tokens)
                    if num_added_tokens > 0:
                        logger.info(f"Added {num_added_tokens} special tokens to {model_name} tokenizer")
                        # Resize the model's embedding matrix to accommodate new tokens
                        model._first_module().auto_model.resize_token_embeddings(len(tokenizer))
                    else:
                        logger.info(f"No new tokens added to {model_name} tokenizer (already present)")
                    
                    self.embedders[model_name] = model
                    
                except Exception as e:
                    logger.error(f"Error initializing {model_name}: {e}")
                    logger.info(f"Falling back to all-MiniLM-L6-v2 for {model_name}")
                    
                    # Fallback to a simpler model
                    fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Add special tokens to fallback model
                    tokenizer = fallback_model.tokenizer
                    num_added_tokens = tokenizer.add_tokens(special_tokens)
                    if num_added_tokens > 0:
                        fallback_model._first_module().auto_model.resize_token_embeddings(len(tokenizer))
                    
                    self.embedders[model_name] = fallback_model
        
        logger.info(f"Initialized {len(self.embedders)} embedding models.")
    
    def generate_embeddings(self, force_regenerate: bool = False) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Generate embeddings for each user sequence using different embedding models.
        
        Args:
            force_regenerate: If True, regenerate embeddings even if they exist
        
        Returns:
            Dictionary with embeddings for each representation and model
        """
        if not self.user_sequences:
            logger.warning("User sequences not found. Generating sequences first.")
            self.generate_user_sequences()
            
        if not self.embedders:
            self.initialize_embedding_models()
            
        embeddings = {rep: {} for rep in self.representations}
        
        logger.info("Generating embeddings...")
        
        # For each representation type
        for rep, sequences_by_user in self.user_sequences.items():
            logger.info(f"Processing representation: {self.representations[rep]}")
            
            # Prepare texts for embedding
            usernames = list(sequences_by_user.keys())
            joined_sequences = [self._join_user_sequence(sequences_by_user[username]) for username in usernames]
            
            # Generate embeddings for each model
            for model_name, model in self.embedders.items():
                logger.info(f"Checking embeddings for {model_name} embeddings for {rep}...")
                
                # Check if embeddings already exist and we're not forcing regeneration
                if not force_regenerate and self._embeddings_exist(rep, model_name):
                    logger.info(f"Loading existing embeddings for {rep}-{model_name}")
                    loaded_embeddings = self._load_embeddings(rep, model_name)
                    
                    if loaded_embeddings is not None:
                        # Verify that loaded embeddings contain all current usernames
                        loaded_usernames = set(loaded_embeddings.keys())
                        current_usernames = set(usernames)
                        
                        if loaded_usernames == current_usernames:
                            embeddings[rep][model_name] = loaded_embeddings
                            logger.info(f"Successfully loaded embeddings for {rep}-{model_name}")
                            continue
                        else:
                            logger.warning(f"Username mismatch for {rep}-{model_name}. Regenerating embeddings.")
                
                # Generate new embeddings
                logger.info(f"Generating new {model_name} embeddings for {rep}...")
                
                if model_name == 'tfidf':
                    # Fit and transform for TF-IDF
                    embeddings_matrix = model.fit_transform(joined_sequences).toarray()
                else:
                    # For SentenceTransformer models
                    logger.debug(f"Encoding sequences with special tokens for {model_name}")
                    embeddings_matrix = model.encode(joined_sequences, show_progress_bar=False)
                
                # Convert to dictionary format
                embeddings_dict = {
                    username: emb for username, emb in zip(usernames, embeddings_matrix)
                }
                
                # Save embeddings to disk
                self._save_embeddings(embeddings_dict, rep, model_name)
                
                # Store embeddings in memory
                embeddings[rep][model_name] = embeddings_dict
                
                logger.info(f"Generated and saved {len(embeddings_dict)} embeddings with {model_name} for {rep}")
        
        # Store the embeddings
        self.embeddings = embeddings
        
        logger.info("Embedding generation complete.")
        return embeddings
    
    def clear_saved_embeddings(self, representation: str = None, model_name: str = None) -> None:
        """
        Clear saved embedding files.
        
        Args:
            representation: If specified, only clear embeddings for this representation
            model_name: If specified, only clear embeddings for this model
        """
        try:
            if representation and model_name:
                # Clear specific embedding
                embedding_file = self._get_embedding_filename(representation, model_name)
                usernames_file = self._get_usernames_filename(representation, model_name)
                
                if os.path.exists(embedding_file):
                    os.remove(embedding_file)
                if os.path.exists(usernames_file):
                    os.remove(usernames_file)
                    
                logger.info(f"Cleared embeddings for {representation}-{model_name}")
                
            elif representation:
                # Clear all embeddings for a representation
                for model in self.embedding_models:
                    embedding_file = self._get_embedding_filename(representation, model)
                    usernames_file = self._get_usernames_filename(representation, model)
                    
                    if os.path.exists(embedding_file):
                        os.remove(embedding_file)
                    if os.path.exists(usernames_file):
                        os.remove(usernames_file)
                        
                logger.info(f"Cleared all embeddings for representation {representation}")
                
            else:
                # Clear all embeddings
                for rep in self.representations:
                    for model in self.embedding_models:
                        embedding_file = self._get_embedding_filename(rep, model)
                        usernames_file = self._get_usernames_filename(rep, model)
                        
                        if os.path.exists(embedding_file):
                            os.remove(embedding_file)
                        if os.path.exists(usernames_file):
                            os.remove(usernames_file)
                            
                logger.info("Cleared all saved embeddings")
                
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
    
    
    def calculate_dunn_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Dunn Index for cluster validation.
        
        Args:
            X: Input data matrix
            labels: Cluster labels
            
        Returns:
            Dunn Index value
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters <= 1:
            return 0.0
        
        # Calculate pairwise distances between all points
        dist_matrix = 1 - cosine_similarity(X)
        
        # Initialize variables for minimum inter-cluster distance and maximum cluster diameter
        min_inter_cluster_dist = float('inf')
        max_cluster_diameter = 0.0
        
        # Calculate inter-cluster distances and cluster diameters
        for i in range(n_clusters):
            cluster_i_indices = np.where(labels == unique_labels[i])[0]
            
            if len(cluster_i_indices) == 0:
                continue
                
            # Calculate cluster diameter (maximum intra-cluster distance)
            if len(cluster_i_indices) > 1:
                cluster_dists = dist_matrix[np.ix_(cluster_i_indices, cluster_i_indices)]
                cluster_diameter = np.max(cluster_dists)
                max_cluster_diameter = max(max_cluster_diameter, cluster_diameter)
            
            # Calculate minimum inter-cluster distances
            for j in range(i + 1, n_clusters):
                cluster_j_indices = np.where(labels == unique_labels[j])[0]
                
                if len(cluster_j_indices) == 0:
                    continue
                    
                # Compute minimum distance between clusters i and j
                inter_cluster_dists = dist_matrix[np.ix_(cluster_i_indices, cluster_j_indices)]
                min_dist = np.min(inter_cluster_dists)
                min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)
        
        # Calculate Dunn Index
        if max_cluster_diameter == 0:
            return 0.0
        
        dunn_index = min_inter_cluster_dist / max_cluster_diameter
        return dunn_index
        
    def calculate_semantic_coherence_optimized(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Optimized version of semantic coherence calculation.
        
        Args:
            X: Input data matrix
            labels: Cluster labels
            
        Returns:
            Average semantic coherence across clusters
        """
        unique_labels = np.unique(labels)
        
        # Skip noise cluster (-1) in HDBSCAN
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) <= 1:
            return 0.0
        
        # Calculate overall centroid once
        overall_centroid = np.mean(X, axis=0).reshape(1, -1)
        
        within_coherence_sum = 0
        between_coherence_sum = 0
        n_valid_clusters = 0
        
        for label in valid_labels:
            cluster_points = X[labels == label]
            
            if len(cluster_points) <= 1:
                continue
                
            n_valid_clusters += 1
            
            # Calculate cluster centroid
            cluster_centroid = np.mean(cluster_points, axis=0).reshape(1, -1)
            
            # OPTIMIZATION 1: Use vectorized cosine similarity instead of nested loops
            # Calculate pairwise similarities within cluster in one go
            if len(cluster_points) > 1:
                cluster_similarities = cosine_similarity(cluster_points)
                # Get upper triangular part (excluding diagonal) to avoid duplicate pairs
                triu_indices = np.triu_indices(len(cluster_similarities), k=1)
                within_sims = cluster_similarities[triu_indices]
                
                if len(within_sims) > 0:
                    within_coherence_sum += np.mean(within_sims)
            
            # OPTIMIZATION 2: Direct cosine similarity calculation
            between_coherence_sum += cosine_similarity(cluster_centroid, overall_centroid)[0][0]
        
        if n_valid_clusters == 0:
            return 0.0
            
        avg_within = within_coherence_sum / n_valid_clusters
        avg_between = between_coherence_sum / n_valid_clusters
        
        # Semantic coherence = within-cluster coherence / between-cluster coherence
        if avg_between == 0:
            return 0.0
            
        semantic_coherence = avg_within / avg_between
        return semantic_coherence

        
    def run_clustering(self, n_clusters: int = 5) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
        """
        Run clustering algorithms on the embeddings.
        
        Args:
            n_clusters: Number of clusters for algorithms that require this parameter
            
        Returns:
            Dictionary with clustering results
        """
        if not self.embeddings:
            logger.warning("Embeddings not found. Generating embeddings first.")
            self.generate_embeddings()
            
        cluster_results = {}
        
        logger.info(f"Running clustering algorithms with {n_clusters} clusters...")
        
        # For each representation
        for rep in self.representations:
            cluster_results[rep] = {}
            
            # For each embedding model
            for model_name in self.embedders:
                if model_name not in self.embeddings[rep]:
                    logger.warning(f"No embeddings found for {model_name} in {rep}")
                    continue
                    
                cluster_results[rep][model_name] = {}
                
                # Get embeddings for this representation and model
                embeddings_dict = self.embeddings[rep][model_name]
                usernames = list(embeddings_dict.keys())
                X = np.array([embeddings_dict[username] for username in usernames])
                
                # Run each clustering algorithm
                for algo_name, algo_func in self.clustering_algorithms.items():
                    logger.info(f"Running {algo_name} on {rep} with {model_name} embeddings...")
                    
                    try:
                        if algo_name == 'hdbscan':
                            # HDBSCAN doesn't need n_clusters
                            clusterer = algo_func()
                        else:
                            clusterer = algo_func(n_clusters)
                            
                        clusterer.fit(X)
                        labels = clusterer.labels_
                        
                        # Store results with username as key
                        cluster_results[rep][model_name][algo_name] = {
                            username: int(label) for username, label in zip(usernames, labels)
                        }
                        
                        logger.info(f"Completed {algo_name} clustering for {rep} with {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error running {algo_name} on {rep} with {model_name}: {e}")
                        # Continue with other algorithms
        
        # Store clustering results
        self.cluster_results = cluster_results
        
        logger.info("Clustering complete.")
        return cluster_results
    
    def evaluate_clusters(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Evaluate clustering results using multiple metrics.
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.cluster_results:
            logger.warning("Cluster results not found. Running clustering first.")
            self.run_clustering()
            
        evaluation_results = {}
        
        logger.info("Evaluating clustering results...")
        
        # For each representation
        for rep in self.representations:
            evaluation_results[rep] = {}
            
            # For each embedding model
            for model_name in self.embedders:
                if model_name not in self.embeddings[rep] or model_name not in self.cluster_results[rep]:
                    continue
                    
                evaluation_results[rep][model_name] = {}
                
                # Get embeddings for this representation and model
                embeddings_dict = self.embeddings[rep][model_name]
                usernames = list(embeddings_dict.keys())
                X = np.array([embeddings_dict[username] for username in usernames])
                
                # Evaluate each clustering algorithm
                for algo_name, cluster_assignments in self.cluster_results[rep][model_name].items():
                    logger.info(f"Evaluating {algo_name} clusters for {rep} with {model_name}...")
                    
                    # Get labels in the same order as X
                    labels = np.array([cluster_assignments[username] for username in usernames])
                    
                    # Skip evaluation if all samples are in one cluster or all are noise
                    if len(np.unique(labels)) <= 1 or np.all(labels == -1):
                        logger.warning(f"Skipping evaluation for {algo_name} with {model_name} - insufficient clusters")
                        evaluation_results[rep][model_name][algo_name] = {
                            "silhouette": 0.0,
                            # "davies_bouldin": float('inf'),
                            "calinski_harabasz": 0.0,
                            "dunn": 0.0,
                            "semantic_coherence": 0.0
                        }
                        continue
                    
                    try:
                        # Calculate evaluation metrics
                        metrics = {}
                        
                        # Filter out noise points for metrics that require it
                        valid_indices = labels != -1
                        X_valid = X[valid_indices]
                        labels_valid = labels[valid_indices]
                        
                        if len(np.unique(labels_valid)) > 1 and len(X_valid) > 1:
                            metrics["silhouette"] = silhouette_score(X_valid, labels_valid)
                            logger.info(f"Silhouette score for {algo_name} with {model_name}: {metrics['silhouette']}")
                            # metrics["davies_bouldin"] = davies_bouldin_score(X_valid, labels_valid)
                            metrics["calinski_harabasz"] = calinski_harabasz_score(X_valid, labels_valid)
                            logger.info(f"Calinski-Harabasz score for {algo_name} with {model_name}: {metrics['calinski_harabasz']}")
                            metrics["dunn"] = self.calculate_dunn_index(X_valid, labels_valid)
                            logger.info(f"Dunn Index for {algo_name} with {model_name}: {metrics['dunn']}")
                            metrics["semantic_coherence"] = self.calculate_semantic_coherence_optimized(X_valid, labels_valid)
                            logger.info(f"Semantic Coherence for {algo_name} with {model_name}: {metrics['semantic_coherence']}")
                        else:
                            # Default values for metrics if evaluation can't be performed
                            metrics = {
                                "silhouette": 0.0,
                                # "davies_bouldin": float('inf'),
                                "calinski_harabasz": 0.0,
                                "dunn": 0.0,
                                "semantic_coherence": 0.0
                            }
                        
                        evaluation_results[rep][model_name][algo_name] = metrics
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {algo_name} for {rep} with {model_name}: {e}")
                        # Use default values if evaluation fails
                        evaluation_results[rep][model_name][algo_name] = {
                            "silhouette": 0.0,
                            # "davies_bouldin": float('inf'),
                            "calinski_harabasz": 0.0,
                            "dunn": 0.0,
                            "semantic_coherence": 0.0
                        }
        
        # Store evaluation results
        self.evaluation_results = evaluation_results
        
        logger.info("Evaluation complete.")
        return evaluation_results
    
    def summarize_results(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Summarize evaluation results across all configurations.
        
        Returns:
            Dictionary with summarized results
        """
        if not self.evaluation_results:
            logger.warning("Evaluation results not found. Evaluating clusters first.")
            self.evaluate_clusters()
            
        summary = {
            "best_representation": {},
            "best_embedding": {},
            "best_clustering": {},
            "top_configurations": []
        }
        
        # Calculate average metrics for each representation, embedding, and clustering
        avg_by_rep = {rep: {"count": 0, "scores": {
            "silhouette": 0, 
            # "davies_bouldin": 0, 
            "calinski_harabasz": 0, 
            "dunn": 0,
             "semantic_coherence": 0
        }} for rep in self.representations}
        
        avg_by_emb = {model: {"count": 0, "scores": {
            "silhouette": 0, 
            # "davies_bouldin": 0, 
            "calinski_harabasz": 0, 
            "dunn": 0, 
            "semantic_coherence": 0
        }} for model in self.embedders}
        
        avg_by_clust = {algo: {"count": 0, "scores": {
            "silhouette": 0, 
            # "davies_bouldin": 0, 
            "calinski_harabasz": 0, 
            "dunn": 0, 
            "semantic_coherence": 0
        }} for algo in self.clustering_algorithms}
        print("avg_by_clust", avg_by_clust)
        
        # Configurations ranked by silhouette score
        all_configs = []
        
        # Aggregate results
        for rep in self.representations:
            for model_name in self.embedders:
                if model_name not in self.evaluation_results.get(rep, {}):
                    continue
                    
                for algo_name, metrics in self.evaluation_results[rep][model_name].items():
                    # Skip invalid configurations
                    if metrics["silhouette"] == 0 and metrics["calinski_harabasz"] == 0:
                        continue
                        
                    # Update averages for representation
                    avg_by_rep[rep]["count"] += 1
                    for metric, value in metrics.items():
                        if metric != "davies_bouldin":  # Higher is better for all except Davies-Bouldin
                            avg_by_rep[rep]["scores"][metric] += value
                        # else:
                        #     # For Davies-Bouldin, lower is better, so use inverse
                        #     avg_by_rep[rep]["scores"][metric] += (1.0 / value) if value > 0 else 0
                    
                    # Update averages for embedding
                    avg_by_emb[model_name]["count"] += 1
                    for metric, value in metrics.items():
                        if metric != "davies_bouldin":
                            avg_by_emb[model_name]["scores"][metric] += value
                        # else:
                        #     avg_by_emb[model_name]["scores"][metric] += (1.0 / value) if value > 0 else 0
                    
                    # Update averages for clustering
                    avg_by_clust[algo_name]["count"] += 1
                    for metric, value in metrics.items():
                        if metric != "davies_bouldin":
                            avg_by_clust[algo_name]["scores"][metric] += value
                        # else:
                        #     avg_by_clust[algo_name]["scores"][metric] += (1.0 / value) if value > 0 else 0
                    
                    # Add to all configurations
                    all_configs.append({
                        "representation": rep,
                        "embedding": model_name,
                        "clustering": algo_name,
                        "metrics": metrics
                    })
        
        # Calculate averages
        for rep, data in avg_by_rep.items():
            if data["count"] > 0:
                for metric in data["scores"]:
                    data["scores"][metric] /= data["count"]
        
        for model_name, data in avg_by_emb.items():
            if data["count"] > 0:
                for metric in data["scores"]:
                    data["scores"][metric] /= data["count"]
        
        for algo_name, data in avg_by_clust.items():
            if data["count"] > 0:
                for metric in data["scores"]:
                    data["scores"][metric] /= data["count"]
        
        # Find best representation
        best_rep = max(avg_by_rep.items(), key=lambda x: x[1]["scores"]["silhouette"] if x[1]["count"] > 0 else -1)
        summary["best_representation"] = {
            "name": best_rep[0],
            "description": self.representations[best_rep[0]],
            "metrics": best_rep[1]["scores"]
        }
        
        # Find best embedding
        best_emb = max(avg_by_emb.items(), key=lambda x: x[1]["scores"]["silhouette"] if x[1]["count"] > 0 else -1)
        summary["best_embedding"] = {
            "name": best_emb[0],
            "metrics": best_emb[1]["scores"]
        }
        
        # Find best clustering
        best_clust = max(avg_by_clust.items(), key=lambda x: x[1]["scores"]["silhouette"] if x[1]["count"] > 0 else -1)
        summary["best_clustering"] = {
            "name": best_clust[0],
            "metrics": best_clust[1]["scores"]
        }
        
        # Sort configurations by silhouette score
        all_configs.sort(key=lambda x: x["metrics"]["silhouette"], reverse=True)
        summary["top_configurations"] = all_configs[:5]  # Top 5 configurations
        
        logger.info("Results summary:")
        logger.info(f"Best representation: {summary['best_representation']['name']} "
                  f"({summary['best_representation']['description']})")
        logger.info(f"Best embedding model: {summary['best_embedding']['name']}")
        logger.info(f"Best clustering algorithm: {summary['best_clustering']['name']}")
        
        return summary
    
    def visualize_results(self, output_dir: str = "results") -> None:
        """
        Visualize the clustering and evaluation results.
        
        Args:
            output_dir: Directory to save visualization results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not self.evaluation_results:
            logger.warning("No evaluation results found. Running evaluation first.")
            self.evaluate_clusters()
            
        logger.info("Generating visualizations...")
        
        # 1. Heatmap of silhouette scores across representations and embedding models
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        reps = list(self.representations.keys())
        embs = list(self.embedders.keys())
        
        # Average silhouette scores across clustering algorithms
        silhouette_matrix = np.zeros((len(reps), len(embs)))
        
        for i, rep in enumerate(reps):
            for j, emb in enumerate(embs):
                if rep in self.evaluation_results and emb in self.evaluation_results[rep]:
                    scores = [metrics["silhouette"] for metrics in self.evaluation_results[rep][emb].values()]
                    silhouette_matrix[i, j] = np.mean(scores) if scores else 0
                    
        # Plot heatmap
        sns.heatmap(silhouette_matrix, annot=True, cmap="YlGnBu", xticklabels=embs, yticklabels=reps)
        plt.title("Average Silhouette Score by Representation and Embedding Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "silhouette_heatmap.png"))
        plt.close()
        
        # 2. Bar chart of best configurations
        summary = self.summarize_results()
        top_configs = summary["top_configurations"]
        
        if top_configs:
            plt.figure(figsize=(14, 8))
            
            # Prepare data
            config_names = [f"{c['representation']}-{c['embedding']}-{c['clustering']}" for c in top_configs]
            silhouette_scores = [c["metrics"]["silhouette"] for c in top_configs]
            semantic_scores = [c["metrics"]["semantic_coherence"] for c in top_configs]
            
            # Plot
            x = np.arange(len(config_names))
            width = 0.35
            
            plt.bar(x - width/2, silhouette_scores, width, label="Silhouette Score")
            plt.bar(x + width/2, semantic_scores, width, label="Semantic Coherence")
            
            plt.xlabel("Configuration")
            plt.ylabel("Score")
            plt.title("Top Performing Configurations")
            plt.xticks(x, config_names, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_configurations.png"))
            plt.close()
            
        # 3. Radar chart for comparing representations
        metrics = ["silhouette", "calinski_harabasz",'dunn', "semantic_coherence"]
        metric_names = ["Silhouette", "Calinski-Harabasz",'Dunn Index', "Semantic Coherence"]
        
        # Calculate average score for each representation and metric
        rep_scores = {}
        for rep in self.representations:
            rep_scores[rep] = {metric: 0 for metric in metrics}
            count = 0
            
            if rep in self.evaluation_results:
                for emb in self.evaluation_results[rep]:
                    for algo in self.evaluation_results[rep][emb]:
                        count += 1
                        for metric in metrics:
                            rep_scores[rep][metric] += self.evaluation_results[rep][emb][algo][metric]
            
            if count > 0:
                for metric in metrics:
                    rep_scores[rep][metric] /= count
        
        # Normalize scores for radar chart
        max_scores = {metric: max(rep_scores[rep][metric] for rep in rep_scores) for metric in metrics}
        normalized_scores = {}
        
        for rep in rep_scores:
            normalized_scores[rep] = {metric: rep_scores[rep][metric] / max_scores[metric] 
                                    if max_scores[metric] > 0 else 0 for metric in metrics}
        
        # Plot radar chart
        plt.figure(figsize=(10, 10))
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], metric_names)
        
        for rep, scores in normalized_scores.items():
            values = [scores[metric] for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=f"{rep}: {self.representations[rep]}")
            ax.fill(angles, values, alpha=0.1)
        
        plt.title("Representation Comparison")
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "representation_radar.png"))
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif hasattr(o, 'tolist'):  # handles numpy arrays and tensors
            return o.tolist()
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    def export_results(self, output_dir: str = "results") -> None:
        """
        Export clustering and evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        def convert(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.int32, np.int64)):
                return int(o)
            elif hasattr(o, 'tolist'):  # handles numpy arrays and tensors
                return o.tolist()
            raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
            
        # Export evaluation results
        if self.evaluation_results:
            with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
                json.dump(self.evaluation_results, f, indent=2, default=convert)
                
        # Export summary
        summary = self.summarize_results()
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=convert)
            
        # Export cluster assignments for the best configuration
        if summary["top_configurations"]:
            best_config = summary["top_configurations"][0]
            rep = best_config["representation"]
            emb = best_config["embedding"]
            algo = best_config["clustering"]
            
            if (rep in self.cluster_results and 
                emb in self.cluster_results[rep] and 
                algo in self.cluster_results[rep][emb]):
                
                cluster_assignments = self.cluster_results[rep][emb][algo]
                
                # Convert to list for JSON serialization
                cluster_list = [{"username": username, "cluster": int(cluster)} 
                               for username, cluster in cluster_assignments.items()]
                
                with open(os.path.join(output_dir, "best_clusters.json"), "w") as f:
                    json.dump({
                        "configuration": {
                            "representation": rep,
                            "embedding": emb,
                            "clustering": algo
                        },
                        "clusters": cluster_list
                    }, f, indent=2)
        
        logger.info(f"Results exported to {output_dir}")

    def close(self) -> None:
        """Close database connection and release resources."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")


def main():
    """Main function to run the user sequence analysis pipeline."""
    # Database configuration
    db_config = {
        "host": "localhost",
        "database": "forum_db",
        "user": "postgres",
        "password": "password"  # Replace with actual password or use environment variables
    }
    
    # Initialize the analysis pipeline
    analyzer = UserSequenceAnalysis(db_config)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Generate user sequences
        analyzer.generate_user_sequences()
        
        # Generate embeddings
        analyzer.generate_embeddings()
        
        # Run clustering
        analyzer.run_clustering(n_clusters=5)  # Adjust number of clusters as needed
        
        # Evaluate clustering
        analyzer.evaluate_clusters()
        
        # Visualize results
        analyzer.visualize_results()
        
        # Export results
        analyzer.export_results()
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise
    finally:
        # Close connections
        analyzer.close()


if __name__ == "__main__":
    
    main()
