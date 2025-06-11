import argparse
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from user_sequence_analysis import UserSequenceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_experiment(args):
    """Setup and configure the experiment based on command line arguments"""
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    config = vars(args)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return experiment_dir

def run_experiment(args):
    """Run the user sequence representation experiment"""
    # Setup experiment directory
    experiment_dir = setup_experiment(args)
    logger.info(f"Starting experiment, results will be saved to {experiment_dir}")
    
    # Database configuration - consider using environment variables for credentials
    db_config = {
        "host": args.db_host,
        "database": args.db_name,
        "user": args.db_user,
        "password": args.db_password
    }
    
    # Initialize the analysis pipeline with custom embedding models if specified
    embedding_models = args.embedding_models.split(',') if args.embedding_models else None
    analyzer = UserSequenceAnalysis(db_config, embedding_models)
    
    try:
        # Load data
        logger.info("Loading data from database...")
        analyzer.load_data()
        
        # Generate user sequences with the four representation strategies
        logger.info("Generating user sequences with all four representation strategies...")
        user_sequences = analyzer.generate_user_sequences()
        
        # Save sample sequences for inspection
        save_sample_sequences(user_sequences, experiment_dir)
        
        # Generate embeddings with selected models
        logger.info("Generating embeddings with selected models...")
        embeddings = analyzer.generate_embeddings()
        
        # Run clustering algorithms
        logger.info(f"Running clustering algorithms with {args.n_clusters} clusters...")
        cluster_results = analyzer.run_clustering()
        
        # Evaluate clustering results
        logger.info("Evaluating clustering results...")
        evaluation_results = analyzer.evaluate_clusters()
        
        # Create visualizations
        logger.info("Creating result visualizations...")
        analyzer.visualize_results(output_dir=experiment_dir)
        
        # Summarize and export results
        logger.info("Summarizing and exporting results...")
        summary = analyzer.summarize_results()
        analyzer.export_results(output_dir=experiment_dir)
        
        # Generate comprehensive report
        generate_report(analyzer, summary, experiment_dir)
        
        logger.info(f"Experiment completed successfully. Results saved to {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        raise
    finally:
        # Close connections
        analyzer.close()

def save_sample_sequences(user_sequences, output_dir):
    """Save sample user sequences for each representation strategy"""
    sample_dir = os.path.join(output_dir, "sample_sequences")
    os.makedirs(sample_dir, exist_ok=True)
    
    # For each representation strategy, save 5 random user sequences
    for rep, user_dict in user_sequences.items():
        usernames = list(user_dict.keys())
        if len(usernames) > 5:
            sample_users = np.random.choice(usernames, 5, replace=False)
        else:
            sample_users = usernames
            
        samples = {user: user_dict[user] for user in sample_users}
        
        with open(os.path.join(sample_dir, f"{rep}_samples.json"), "w") as f:
            json.dump(samples, f, indent=2)
    
    logger.info(f"Sample sequences saved to {sample_dir}")

def generate_report(analyzer, summary, output_dir):
    """Generate a comprehensive report of the experiment results"""
    report_path = os.path.join(output_dir, "experiment_report.md")
    
    with open(report_path, "w") as f:
        f.write("# User Sequence Representation Experiment Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write("This experiment evaluates four different user sequence representation strategies ")
        f.write("using various embedding models and clustering algorithms.\n\n")
        
        # Representation strategies
        f.write("## Representation Strategies\n\n")
        for rep, desc in analyzer.representations.items():
            f.write(f"- **{rep}**: {desc}\n")
        f.write("\n")
        
        # Dataset statistics
        f.write("## Dataset Statistics\n\n")
        if analyzer.data:
            f.write(f"- **Users**: {len(analyzer.data['members'])}\n")
            f.write(f"- **Threads**: {len(analyzer.data['threads'])}\n")
            f.write(f"- **Posts**: {len(analyzer.data['posts'])}\n")
            f.write(f"- **Interactions**: {len(analyzer.data['interactions'])}\n")
        f.write("\n")
        
        # Best results
        f.write("## Best Results\n\n")
        f.write("### Best Representation Strategy\n\n")
        f.write(f"**{summary['best_representation']['name']}**: {summary['best_representation']['description']}\n\n")
        f.write("Metrics:\n")
        for metric, value in summary['best_representation']['metrics'].items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("### Best Embedding Model\n\n")
        f.write(f"**{summary['best_embedding']['name']}**\n\n")
        f.write("Metrics:\n")
        for metric, value in summary['best_embedding']['metrics'].items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("### Best Clustering Algorithm\n\n")
        f.write(f"**{summary['best_clustering']['name']}**\n\n")
        f.write("Metrics:\n")
        for metric, value in summary['best_clustering']['metrics'].items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        # Top configurations
        f.write("## Top Configurations\n\n")
        f.write("| Rank | Representation | Embedding | Clustering | Silhouette | D-B Index | C-H Index | Semantic Coherence |\n")
        f.write("|------|---------------|-----------|------------|------------|-----------|-----------|-------------------|\n")
        
        for i, config in enumerate(summary['top_configurations']):
            metrics = config['metrics']
            f.write(f"| {i+1} | {config['representation']} | {config['embedding']} | {config['clustering']} | ")
            f.write(f"{metrics['calinski_harabasz']:.4f} |  ")
            f.write(f"{metrics['semantic_coherence']:.4f} |\n")
            f.write(f"{metrics['dunn']:.4f} |\n")
        f.write("\n")
        
        # Include visualizations
        f.write("## Visualizations\n\n")
        f.write("### Silhouette Score Heatmap\n\n")
        f.write("![Silhouette Score Heatmap](silhouette_heatmap.png)\n\n")
        
        f.write("### Top Configurations Comparison\n\n")
        f.write("![Top Configurations](top_configurations.png)\n\n")
        
        f.write("### Representation Strategy Comparison\n\n")
        f.write("![Representation Radar](representation_radar.png)\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        best_rep = summary['best_representation']['name']
        best_emb = summary['best_embedding']['name']
        best_clust = summary['best_clustering']['name']
        
        f.write("Based on our evaluation metrics, the optimal configuration is:\n\n")
        f.write(f"- **Representation Strategy**: {best_rep} ({analyzer.representations[best_rep]})\n")
        f.write(f"- **Embedding Model**: {best_emb}\n")
        f.write(f"- **Clustering Algorithm**: {best_clust}\n\n")
        
        if best_rep == 'R1':
            f.write("The full representation strategy performed best, suggesting that preserving all ")
            f.write("content details yields more meaningful user clusters than topic summarization.\n")
        elif best_rep == 'R4':
            f.write("The most concise representation (both threads and replies summarized) performed best, ")
            f.write("suggesting that topic-level abstraction captures user characteristics effectively ")
            f.write("while reducing noise from specific content details.\n")
        else:
            f.write("A hybrid representation strategy performed best, suggesting that some content ")
            f.write("is more informative in full form while other content benefits from summarization.\n")
    
    logger.info(f"Experiment report generated: {report_path}")

def main():
    """Parse command line arguments and run the experiment"""
    parser = argparse.ArgumentParser(description="Run user sequence representation experiment")
    
    # Database configuration
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-name", default="cracked", help="Database name")
    parser.add_argument("--db-user", default="", help="Database user")
    parser.add_argument("--db-password", default="", help="Database password")
    
    # Experiment configuration
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--embedding-models", type=str,
                        help="Comma-separated list of embedding models")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(args)

if __name__ == "__main__":
    main()
