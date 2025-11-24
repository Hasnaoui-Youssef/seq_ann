#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def scan_vhdl_file(filepath):
    """
    Scans a VHDL file for entity declarations, package declarations,
    and dependencies (use statements, entity instantiations).
    """
    entities = []
    packages = []
    dependencies = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().lower()
        
        # Find entity declarations
        entities.extend(re.findall(r'entity\s+(\w+)\s+is', content))
        
        # Find package declarations (excluding bodies)
        packages.extend(re.findall(r'package\s+(\w+)\s+is', content))
        
        # Find dependencies via 'use work.X.all'
        dependencies.extend(re.findall(r'use\s+work\.(\w+)', content))
        
        # Find dependencies via entity instantiation 'entity work.X'
        dependencies.extend(re.findall(r'entity\s+work\.(\w+)', content))
        
    return {
        'file': filepath,
        'provides': set(entities + packages),
        'requires': set(dependencies)
    }

def build_dependency_graph(src_dir):
    """
    Builds a directed graph of dependencies from VHDL files in src_dir.
    """
    files_data = {}
    provides_map = {}
    
    # Scan all .vhd files
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.vhd'):
                filepath = os.path.join(root, file)
                data = scan_vhdl_file(filepath)
                files_data[filepath] = data
                
                for item in data['provides']:
                    provides_map[item] = filepath
                    
    # Build graph
    G = nx.DiGraph()
    
    for filepath, data in files_data.items():
        G.add_node(filepath, label=os.path.basename(filepath))
        
        for req in data['requires']:
            if req in provides_map:
                provider = provides_map[req]
                if provider != filepath: # Avoid self-loops
                    G.add_edge(provider, filepath)
            else:
                # Dependency not found in scanned files (could be external or std lib)
                pass
                
    return G

import subprocess
import sys

def generate_visual_graph(G, output_path):
    """
    Generates a visual representation of the dependency graph using Graphviz (dot).
    This provides a superior top-down hierarchical layout compared to Matplotlib.
    """
    dot_content = ["digraph G {"]
    dot_content.append('    rankdir=TB;') # Top-to-Bottom layout
    dot_content.append('    node [shape=box, style=filled, fillcolor=lightblue, fontname="Helvetica"];')
    dot_content.append('    edge [color=gray50];')
    dot_content.append('')
    
    # Add nodes
    for node in G.nodes():
        label = os.path.basename(node)
        dot_content.append(f'    "{node}" [label="{label}"];')
        
    # Add edges
    # G.edges() are (Provider, Consumer)
    # User wants Top-Down view where Provider (e.g. types) is at the Top.
    # Consumer (which depends on Provider) should be underneath.
    # So we draw edge: Provider -> Consumer
    for u, v in G.edges():
        dot_content.append(f'    "{u}" -> "{v}";')
        
    dot_content.append("}")
    
    dot_source = "\n".join(dot_content)
    
    try:
        # Run dot command
        process = subprocess.run(
            ['dot', '-Tpng', '-o', output_path],
            input=dot_source.encode('utf-8'),
            check=True,
            capture_output=True
        )
        print(f"Dependency graph saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Graphviz dot: {e.stderr.decode('utf-8')}")
        print("Ensure 'graphviz' is installed (sudo apt install graphviz).")
    except FileNotFoundError:
        print("Error: 'dot' command not found. Please install Graphviz.")

def get_sorted_files(G):
    """
    Returns a list of files sorted by dependency order (topological sort).
    """
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("ERROR: Circular dependency detected!")
        # Find cycle
        try:
            cycle = nx.find_cycle(G)
            print(f"Cycle: {cycle}")
        except:
            pass
        return list(G.nodes()) # Return unsorted as fallback

def main():
    parser = argparse.ArgumentParser(description='Scan VHDL dependencies')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--graph', default='dependencies.png', help='Output path for dependency graph image')
    parser.add_argument('--list', action='store_true', help='Output sorted file list for Makefile')
    
    args = parser.parse_args()
    
    G = build_dependency_graph(args.src)
    
    if args.graph:
        generate_visual_graph(G, args.graph)
        
    if args.list:
        sorted_files = get_sorted_files(G)
        print(' '.join(sorted_files))

if __name__ == '__main__':
    main()
