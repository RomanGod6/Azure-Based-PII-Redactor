#!/usr/bin/env python3
"""
Batch Processing Script for PII Redactor
Process multiple CSV files in a directory
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple
import json
import time
from dotenv import load_dotenv

# Import the Azure PII detector
from azure_pii_detector import AzurePIIDetector, LocalPIIDetector


class BatchProcessor:
    """Batch processor for multiple CSV files"""
    
    def __init__(self, use_azure: bool = True):
        """
        Initialize batch processor
        
        Args:
            use_azure: Whether to use Azure AI (True) or local detection (False)
        """
        load_dotenv()
        
        if use_azure:
            endpoint = os.getenv("AZURE_ENDPOINT")
            key = os.getenv("AZURE_KEY")
            
            if endpoint and key:
                self.detector = AzurePIIDetector(endpoint, key)
                self.mode = "Azure AI"
            else:
                print("⚠️  Azure credentials not found, using local detection")
                self.detector = LocalPIIDetector()
                self.mode = "Local"
        else:
            self.detector = LocalPIIDetector()
            self.mode = "Local"
        
        self.results = []
        self.total_cost = 0.0
        
    def process_file(self, file_path: Path) -> Dict:
        """
        Process a single CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with processing results
        """
        print(f"\n📄 Processing: {file_path.name}")
        start_time = time.time()
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            rows, cols = df.shape
            print(f"   Shape: {rows} rows × {cols} columns")
            
            # Process based on detector type
            if isinstance(self.detector, AzurePIIDetector):
                # Estimate cost
                estimated_cost = self.detector.estimate_cost(df)
                print(f"   Estimated cost: ${estimated_cost:.4f}")
                
                # Process with Azure
                redacted_df, stats = self.detector.detect_and_redact_dataframe(df)
                actual_cost = stats['cost']
                entities_found = stats['entities_found']
                cells_with_pii = stats['cells_with_pii']
            else:
                # Process with local detector
                redacted_df = self.detector.redact_dataframe(df)
                actual_cost = 0.0
                entities_found = {}
                cells_with_pii = 0
            
            # Generate output filename
            output_name = f"{file_path.stem}_redacted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = file_path.parent / output_name
            
            # Save redacted file
            redacted_df.to_csv(output_path, index=False)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update total cost
            self.total_cost += actual_cost
            
            # Create result
            result = {
                'file': file_path.name,
                'status': 'Success',
                'rows': rows,
                'columns': cols,
                'cells_with_pii': cells_with_pii,
                'entities_found': entities_found,
                'cost': actual_cost,
                'duration': duration,
                'output_file': str(output_path)
            }
            
            print(f"   ✅ Completed in {duration:.2f} seconds")
            print(f"   💾 Saved to: {output_path.name}")
            
            if isinstance(self.detector, AzurePIIDetector):
                print(f"   🔍 Found {cells_with_pii} cells with PII")
                print(f"   💰 Cost: ${actual_cost:.4f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                'file': file_path.name,
                'status': 'Failed',
                'error': str(e)
            }
    
    def process_directory(self, directory: Path, pattern: str = "*.csv") -> List[Dict]:
        """
        Process all CSV files in a directory
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            
        Returns:
            List of processing results
        """
        csv_files = list(directory.glob(pattern))
        
        if not csv_files:
            print(f"No files matching '{pattern}' found in {directory}")
            return []
        
        print(f"\n🔍 Found {len(csv_files)} CSV files to process")
        print(f"🤖 Using {self.mode} for PII detection")
        print("=" * 60)
        
        for file_path in csv_files:
            # Skip already redacted files
            if '_redacted_' in file_path.name:
                print(f"\n⏭️  Skipping already redacted file: {file_path.name}")
                continue
            
            result = self.process_file(file_path)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self, output_dir: Path = None) -> str:
        """
        Generate processing report
        
        Args:
            output_dir: Directory to save report (None = current directory)
            
        Returns:
            Path to report file
        """
        if not self.results:
            print("No results to report")
            return None
        
        output_dir = output_dir or Path.cwd()
        report_file = output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate summary statistics
        successful = [r for r in self.results if r.get('status') == 'Success']
        failed = [r for r in self.results if r.get('status') == 'Failed']
        
        total_rows = sum(r.get('rows', 0) for r in successful)
        total_cells_with_pii = sum(r.get('cells_with_pii', 0) for r in successful)
        total_duration = sum(r.get('duration', 0) for r in successful)
        
        # Aggregate entities found
        all_entities = {}
        for r in successful:
            for entity, count in r.get('entities_found', {}).items():
                all_entities[entity] = all_entities.get(entity, 0) + count
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'summary': {
                'files_processed': len(successful),
                'files_failed': len(failed),
                'total_rows': total_rows,
                'total_cells_with_pii': total_cells_with_pii,
                'total_cost': self.total_cost,
                'total_duration_seconds': total_duration,
                'entities_found': all_entities
            },
            'details': self.results
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING REPORT")
        print("=" * 60)
        print(f"Files processed: {len(successful)}")
        print(f"Files failed: {len(failed)}")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Cells with PII found: {total_cells_with_pii:,}")
        print(f"Total cost: ${self.total_cost:.4f}")
        print(f"Total time: {total_duration:.2f} seconds")
        print(f"\nReport saved to: {report_file}")
        
        return str(report_file)
    
    def print_summary(self):
        """Print processing summary"""
        if not self.results:
            return
        
        print("\n" + "=" * 60)
        print("📈 PROCESSING SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            status = "✅" if result['status'] == 'Success' else "❌"
            print(f"\n{status} {result['file']}")
            
            if result['status'] == 'Success':
                print(f"   Output: {Path(result['output_file']).name}")
                if 'cost' in result:
                    print(f"   Cost: ${result['cost']:.4f}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")


def main():
    """Main function for batch processing"""
    parser = argparse.ArgumentParser(
        description='Batch process CSV files for PII redaction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files in current directory
  python batch_process.py
  
  # Process specific directory
  python batch_process.py --dir /path/to/csvs
  
  # Use local detection instead of Azure
  python batch_process.py --local
  
  # Process only files matching pattern
  python batch_process.py --pattern "zendesk_*.csv"
  
  # Save report to specific directory
  python batch_process.py --output /path/to/reports
        """
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='.',
        help='Directory containing CSV files (default: current directory)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.csv',
        help='File pattern to match (default: *.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for reports (default: same as input)'
    )
    
    parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='Use local PII detection instead of Azure AI'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show files that would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    input_dir = Path(args.dir)
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        csv_files = list(input_dir.glob(args.pattern))
        csv_files = [f for f in csv_files if '_redacted_' not in f.name]
        
        print(f"🔍 Would process {len(csv_files)} files:")
        for f in csv_files:
            print(f"   - {f.name}")
        sys.exit(0)
    
    # Create processor
    processor = BatchProcessor(use_azure=not args.local)
    
    # Process files
    results = processor.process_directory(input_dir, args.pattern)
    
    if results:
        # Generate report
        output_dir = Path(args.output) if args.output else input_dir
        processor.generate_report(output_dir)
        
        # Print summary
        processor.print_summary()
    else:
        print("No files were processed")


if __name__ == "__main__":
    main()
