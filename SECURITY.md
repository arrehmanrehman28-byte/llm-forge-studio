# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover security vulnerability:

1. **DO NOT** open public issue
2. Email maintainers directly or use GitHub private vulnerability reporting
3. Include description, steps to reproduce, potential impact
4. Allow 90 days for fix before public disclosure

### What to report:
- Model injection attacks
- Data leakage
- API security issues
- Dependency vulnerabilities

### What NOT to report:
- Training data quality (use data validation)
- Model hallucinations (expected LLM behavior)

## Security Best Practices for Users

- **Never** commit HF tokens, API keys - Use `.env`
- **Validate** datasets before training - Prevent injection
- **Use** SafeTensors (not pickle) - Prevent code execution
- **Scan** exported GGUF files
- **Isolate** training in Docker for untrusted data

## Dependencies

We monitor dependencies via Dependabot. Security updates via PR.

Thank you for keeping LLM Forge secure! 🔒
