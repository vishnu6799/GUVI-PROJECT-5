# 🚀 AWS Deployment Guide — Depression Risk Predictor

This guide covers two deployment options:
1. **AWS EC2** (manual, full control)
2. **AWS App Runner** (managed, simplest)

---

## Option A: AWS App Runner (Recommended — Easiest)

### Step 1: Push image to Amazon ECR

```bash
# Set your variables
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=depression-predictor

# Create ECR repository
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push
docker build -t $ECR_REPO .
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

### Step 2: Create App Runner Service

Go to [AWS App Runner Console](https://console.aws.amazon.com/apprunner) → **Create service**:
- Source: **Container registry** → Amazon ECR
- Image URI: `<account>.dkr.ecr.<region>.amazonaws.com/depression-predictor:latest`
- Port: `8501`
- CPU: 1 vCPU, Memory: 2 GB
- Click **Create & Deploy**

Your app will be live at: `https://<random>.awsapprunner.com`

---

## Option B: AWS EC2

### Step 1: Launch EC2 Instance

```bash
# Launch t3.small (sufficient for this app)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \  # Amazon Linux 2023 us-east-1
  --instance-type t3.small \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=depression-app}]'
```

**Security Group rules** (inbound):
- Port 22 (SSH) from your IP
- Port 8501 (Streamlit) from 0.0.0.0/0

### Step 2: Install Docker on EC2

```bash
ssh -i your-key.pem ec2-user@<PUBLIC_IP>

sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -aG docker ec2-user
# Log out and back in
```

### Step 3: Deploy

```bash
# Option 1: Transfer files directly
scp -i your-key.pem -r ./depression_app ec2-user@<PUBLIC_IP>:~/

# On the EC2 instance:
cd ~/depression_app
docker build -t depression-app .
docker run -d -p 8501:8501 --restart unless-stopped --name depression-app depression-app

# Verify
docker ps
curl http://localhost:8501/_stcore/health
```

App accessible at: `http://<EC2_PUBLIC_IP>:8501`

### Step 4 (Optional): Add HTTPS with nginx

```bash
sudo yum install -y nginx certbot python3-certbot-nginx

# /etc/nginx/conf.d/streamlit.conf
cat <<EOF | sudo tee /etc/nginx/conf.d/streamlit.conf
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

sudo nginx -t && sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

---

## Option C: Local Run (Testing)

```bash
cd depression_app
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

---

## CI/CD with GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECR + App Runner

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - uses: aws-actions/amazon-ecr-login@v1
      - name: Build and push
        run: |
          docker build -t depression-predictor .
          docker tag depression-predictor:latest ${{ secrets.ECR_URI }}:latest
          docker push ${{ secrets.ECR_URI }}:latest
      - name: Update App Runner
        run: |
          aws apprunner start-deployment --service-arn ${{ secrets.APP_RUNNER_ARN }}
```

---

## Cost Estimate

| Service | Config | Monthly Cost |
|---------|--------|-------------|
| EC2 t3.small | On-demand | ~$15 |
| EC2 t3.small | Spot | ~$5 |
| App Runner | 1 vCPU / 2GB | ~$20 (+ $5 provisioned) |
| ECR | 500MB storage | ~$0.05 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8501 unreachable | Check EC2 security group inbound rules |
| Container exits immediately | `docker logs depression-app` for errors |
| Memory error | Upgrade to t3.medium |
| Streamlit websocket issues | Add nginx proxy headers (see Step 4) |
