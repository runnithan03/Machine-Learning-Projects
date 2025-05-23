import boto3
import pandas as pd
from io import StringIO

# Initialize AWS clients
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# Function to upload data to S3
def upload_to_s3(data, bucket_name, file_name):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())

# Function to start SageMaker training job
def start_sagemaker_training(job_name, role_arn, bucket_name, train_data, output_path):
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': '123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
            'TrainingInputMode': 'File'
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{bucket_name}/{train_data}'
                    }
                }
            }
        ],
        OutputDataConfig={
            'S3OutputPath': f's3://{bucket_name}/{output_path}'
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 86400
        }
    )

# Main function
def main():
    # Load your data
    data = pd.read_csv('portfolio_data.csv')
    
    # Upload data to S3
    upload_to_s3(data, 'your-bucket-name', 'portfolio_data.csv')
    
    # Start SageMaker training job
    start_sagemaker_training(
        'portfolio-optimization-job',
        'arn:aws:iam::123456789012:role/SageMakerRole',
        'your-bucket-name',
        'portfolio_data.csv',
        'model_output'
    )

if __name__ == "__main__":
    main()
