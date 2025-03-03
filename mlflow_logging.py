import dagshub
dagshub.init(repo_owner='Prathmesh-22', repo_name='my-first-repo', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)