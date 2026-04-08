# AWS Lambda Deployment Guide

This guide covers deploying the tg_summarizer to AWS Lambda using two approaches:
1. **Manual deployment** using the AWS CLI and deployment package
2. **Infrastructure as Code** using AWS SAM

## Prerequisites

- AWS CLI v2 installed and configured with appropriate credentials
- Python 3.12 installed locally
- Access to all required Telegram credentials and OpenAI API key
- (Optional) AWS SAM CLI for infrastructure-as-code deployment

## Option 1: Manual Deployment

### Step 1: Build the deployment package

```bash
./build_lambda_package.sh --clean
```

This creates `tg_summarizer_lambda.zip` with all dependencies.

### Step 2: Create an S3 bucket for state persistence

```bash
aws s3 mb s3://your-tg-summarizer-state-bucket --region us-east-1
```

### Step 3: Create IAM role for Lambda

```bash
cat > lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name tg-summarizer-lambda-role \
  --assume-role-policy-document file://lambda-trust-policy.json
```

### Step 4: Attach IAM policies

```bash
# Basic Lambda execution logs
aws iam attach-role-policy \
  --role-name tg-summarizer-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# S3 access for state persistence
cat > s3-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-tg-summarizer-state-bucket",
        "arn:aws:s3:::your-tg-summarizer-state-bucket/*"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name tg-summarizer-lambda-role \
  --policy-name s3-state-access \
  --policy-document file://s3-policy.json
```

### Step 5: Create the Lambda function

```bash
aws lambda create-function \
  --function-name tg-summarizer \
  --runtime python3.12 \
  --handler lambda_handler.handler \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/tg-summarizer-lambda-role \
  --zip-file fileb://tg_summarizer_lambda.zip \
  --timeout 180 \
  --memory-size 512 \
  --environment "Variables={TELEGRAM_API_ID=your_api_id,TELEGRAM_API_HASH=your_api_hash,TELEGRAM_BOT_TOKEN=your_bot_token,TARGET_CHANNEL=@dstgsum,SOURCE_CHANNELS=@channel1,@channel2,OPENAI_API_KEY=your_openai_key,OPENAI_MODEL=gpt-4o-mini,STATE_S3_BUCKET=your-tg-summarizer-state-bucket,STATE_S3_PREFIX=tg_summarizer/prod}"
```

### Step 6: Set up EventBridge trigger

```bash
# Create EventBridge rule for daily execution at 9:00 UTC
aws events put-rule \
  --name tg-summarizer-daily-run \
  --schedule-expression "cron(0 9 * * ? *)" \
  --description "Daily trigger for tg_summarizer"

# Grant EventBridge permission to invoke Lambda
cat > eventbridge-permission.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "events.amazonaws.com"
      },
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:tg-summarizer",
      "Condition": {
        "ArnLike": {
          "AWS:SourceArn": "arn:aws:events:us-east-1:YOUR_ACCOUNT_ID:rule/tg-summarizer-daily-run"
        }
      }
    }
  ]
}
EOF

aws lambda add-permission \
  --function-name tg-summarizer \
  --statement-id eventbridge-trigger \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:YOUR_ACCOUNT_ID:rule/tg-summarizer-daily-run

# Link rule to function with event payload
aws events put-targets \
  --rule tg-summarizer-daily-run \
  --targets "Id"="tg-summarizer-target","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:tg-summarizer","Input"="{\"send_message\": true, \"save_changes\": true, \"include_today_processed_groups\": false, \"include_today_processed_messages\": false}"
```

### Step 7: Test the function

```bash
aws lambda invoke \
  --function-name tg-summarizer \
  --payload '{"send_message": false, "save_changes": true, "include_today_processed_groups": false, "include_today_processed_messages": false}' \
  response.json

cat response.json
```

## Option 2: AWS SAM Deployment

### Step 1: Install AWS SAM CLI

```bash
# macOS
brew install aws-sam-cli

# Or follow official guide: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
```

### Step 2: Configure environment

Create a SAM configuration file:

```bash
cp samconfig.toml.example samconfig.toml
# Edit samconfig.toml with your parameters
```

### Step 3: Build and deploy

```bash
# Build the application
sam build

# Deploy (first time, guided setup)
sam deploy --guided

# Subsequent deployments
sam deploy
```

Example `samconfig.toml`:

```toml
version = 0.1

[default.deploy.parameters]
stack_name = "tg-summarizer"
region = "us-east-1"
confirm_changeset = true
capabilities = "CAPABILITY_IAM CAPABILITY_NAMED_IAM"
parameter_overrides = [
    "TelegramApiId=12345678",
    "TelegramApiHash=your_api_hash",
    "TelegramBotToken=your_bot_token",
    "OpenAIApiKey=sk-your_key",
    "TargetChannel=@dstgsum",
    "SourceChannels=@channel1,@channel2",
    "StateBucket=your-state-bucket",
    "ScheduleExpression=cron(0 9 * * ? *)",
    "EnableSsmSend=true"
]
```

### Step 4: Update function code

After initial deployment, to update just the code:

```bash
sam build
sam deploy --no-confirm-changeset
```

## Updating the function

### Using AWS CLI

```bash
./build_lambda_package.sh --clean

aws lambda update-function-code \
  --function-name tg-summarizer \
  --zip-file fileb://tg_summarizer_lambda.zip
```

### Using AWS SAM

```bash
sam build
sam deploy
```

### Environment variable updates

```bash
aws lambda update-function-configuration \
  --function-name tg-summarizer \
  --environment "Variables={TELEGRAM_API_ID=new_id,TELEGRAM_API_HASH=new_hash,...}"
```

## Monitoring

### CloudWatch Logs

```bash
# View recent logs
aws logs tail /aws/lambda/tg-summarizer --follow

# Filter for errors
aws logs tail /aws/lambda/tg-summarizer --filter-pattern "ERROR"
```

### CloudWatch Metrics

Monitor these metrics in CloudWatch Console:
- `Duration` - execution time
- `Errors` - error count
- `Invocations` - number of invocations
- `Throttles` - number of throttles

### S3 state verification

```bash
# Check if state files are being saved
aws s3 ls s3://your-tg-summarizer-state-bucket/tg_summarizer/prod/
```

## Troubleshooting

### Function timeout
- Increase `Timeout` in Lambda configuration (max 15 minutes)
- Check CloudWatch logs for progress
- Consider reducing `SOURCE_CHANNELS` count

### Module import errors
- Ensure `build_lambda_package.sh` was run successfully
- Verify all dependencies are in `requirements.txt`
- Check Python version compatibility (3.12)

### S3 sync errors
- Verify IAM permissions for the Lambda execution role
- Check `STATE_S3_BUCKET` and `STATE_S3_PREFIX` values
- Ensure S3 bucket exists and is accessible

### Telegram connection errors
- Verify `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` are correct
- Check network connectivity from Lambda to Telegram API
- Consider Lambda VPC configuration if using private endpoints

## Cost optimization

- Use `gpt-4o-mini` as the OpenAI model (cheapest option)
- Set appropriate schedule to avoid unnecessary invocations
- Use `send_message=false` for testing runs
- Monitor CloudWatch metrics to right-size memory allocation

## Security best practices

- Store sensitive credentials in AWS Secrets Manager or SSM Parameter Store
- Use IAM roles with least-privilege permissions
- Enable Lambda function URL authentication if exposing endpoints
- Rotate API keys regularly
- Monitor CloudTrail for Lambda API activity
