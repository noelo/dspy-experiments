```
helm upgrade -i mlflow-server strangiato/mlflow-server --set openshiftOauth.enableBearerTokenAccess=true --set objectStorage.objectBucketClaim.enabled=false --set objectStorage.s3SecretAccessKey=minio123 --set objectStorage.s3EndpointUrl=https://minio-api-mlflow.apps.snoai.bohereen.com --set objectStorage.s3AccessKeyId=minio

```
