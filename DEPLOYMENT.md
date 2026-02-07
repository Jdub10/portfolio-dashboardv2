# Production Deployment Guide

## Pre-Deployment Checklist

- [ ] All secrets stored in environment variables
- [ ] Requirements.txt tested and frozen
- [ ] Error handling tested with bad data
- [ ] Authentication verified
- [ ] Performance tested with full dataset
- [ ] Logging configured appropriately
- [ ] HTTPS/SSL certificate ready
- [ ] Backup strategy defined

## Deployment Options

### 1. Streamlit Cloud (Recommended for Prototype)

**Pros**: Free tier, easy setup, automatic scaling
**Cons**: Public unless paid plan, limited resources

**Steps**:
```bash
# 1. Push to GitHub (private repo recommended)
git init
git add .
git commit -m "Initial dashboard deployment"
git remote add origin your-repo-url
git push -u origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repository
# 4. Add secrets in Advanced Settings
# 5. Deploy!
```

**Secrets Configuration**:
- Navigate to App Settings → Secrets
- Paste contents of your secrets.toml
- Save and restart app

### 2. Docker Container (Recommended for Production)

**Pros**: Full control, portable, scalable
**Cons**: Requires infrastructure management

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY portfolio_dashboard.py .

# Create secrets directory
RUN mkdir -p .streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "portfolio_dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
```

**Build and Run**:
```bash
# Build image
docker build -t portfolio-dashboard .

# Run container
docker run -p 8501:8501 \
  -e PASSWORD="your-password" \
  portfolio-dashboard

# Or with docker-compose (see below)
docker-compose up -d
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PASSWORD=${PASSWORD}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. AWS (EC2 or ECS)

**EC2 Deployment**:
```bash
# 1. Launch EC2 instance (t3.small or larger)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip nginx certbot -y

# 4. Clone repository
git clone your-repo-url
cd your-repo

# 5. Install Python packages
pip3 install -r requirements.txt

# 6. Set up environment
export PASSWORD="your-password"

# 7. Run with PM2 (process manager)
sudo npm install -g pm2
pm2 start "streamlit run portfolio_dashboard.py" --name dashboard
pm2 save
pm2 startup

# 8. Configure Nginx reverse proxy
sudo nano /etc/nginx/sites-available/dashboard
```

**Nginx Configuration**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**SSL Certificate**:
```bash
sudo certbot --nginx -d your-domain.com
```

### 4. Google Cloud Run

**Steps**:
```bash
# 1. Install gcloud CLI
# 2. Authenticate
gcloud auth login

# 3. Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/dashboard

# 4. Deploy to Cloud Run
gcloud run deploy dashboard \
  --image gcr.io/your-project/dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars PASSWORD="your-password"
```

## Environment Variables

Set these in your deployment platform:

```bash
# Required
PASSWORD=your-secure-password

# Optional
SHEET_URL=https://docs.google.com/...
DEFAULT_FX_RATE=0.70
CACHE_TTL=600
```

## Monitoring & Maintenance

### Logging
```python
# Already implemented in code
# Check logs with:
streamlit run portfolio_dashboard.py --logger.level=debug

# Docker logs:
docker logs -f container-name

# Cloud platforms: use native logging
```

### Performance Monitoring
- Monitor response times
- Track cache hit rates
- Watch memory usage
- Set up alerts for errors

### Backup Strategy
- Database: Google Sheets (auto-backed)
- Code: Git version control
- Secrets: Secure vault (1Password, AWS Secrets Manager)

### Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
# Streamlit Cloud: auto-deploys
# Docker: docker-compose restart
# PM2: pm2 restart dashboard
```

## Security Best Practices

1. **Never commit secrets**: Use .gitignore
2. **Use HTTPS**: Always in production
3. **Strong passwords**: 16+ characters, mixed case, symbols
4. **Regular updates**: Keep dependencies current
5. **Access logs**: Monitor for suspicious activity
6. **Rate limiting**: Prevent abuse (nginx/cloudflare)
7. **Firewall rules**: Restrict access if internal tool

## Scaling Considerations

### Vertical Scaling
- Increase instance size
- Add more memory/CPU
- Optimize cache settings

### Horizontal Scaling
- Load balancer + multiple instances
- Shared cache (Redis)
- Database for state management

### Performance Tips
- Increase `CACHE_TTL` for stability
- Use CDN for static assets
- Optimize data queries
- Consider batch updates

## Troubleshooting Production Issues

### App won't start
```bash
# Check logs
docker logs container-name

# Verify environment variables
printenv | grep PASSWORD

# Test locally
streamlit run portfolio_dashboard.py
```

### Slow performance
- Check cache TTL settings
- Verify network latency to APIs
- Monitor memory usage
- Review data size

### Authentication failures
- Verify PASSWORD environment variable
- Check secrets configuration
- Clear browser cache/cookies

## Cost Optimization

- **Streamlit Cloud Free**: $0/month (limited resources)
- **Docker on VPS**: $5-20/month (DigitalOcean, Linode)
- **AWS EC2 t3.small**: ~$15/month
- **Google Cloud Run**: Pay-per-request, ~$5-10/month

## Support & Resources

- Streamlit Docs: https://docs.streamlit.io
- Docker Docs: https://docs.docker.com
- AWS Docs: https://aws.amazon.com/documentation/
- Community: Streamlit Forum, Stack Overflow

---

**Remember**: Always test in staging before production deployment!
