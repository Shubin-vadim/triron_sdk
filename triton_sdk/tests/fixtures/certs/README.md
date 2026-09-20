# Test certificates

Self-signed, test-only, valid ~100 years (openssl). `ca.*` signs `server.*` (CN=localhost, SAN localhost/127.0.0.1)
and `client.*` (mTLS). Never use outside tests.
