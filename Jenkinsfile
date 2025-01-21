pipeline {
    agent any

    stages {
        stage('checkout') {
            steps {
                checkout scmGit(branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 'git', url: 'https://github.com/Achref-dot-afk/MLOps-1.git']])
            }
        }
        stage('Initialize') {
            steps {
                script {
                    sh '''
                        apt update && apt install uvicorn && apt install joblib
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Load and Preprocess Data') {
            steps {
                script {
                    // Run data loading script
                    sh '''
                          . venv/bin/activate 
                          python3 data_loading.py
                    '''
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    // Run model training script
                    sh ''' 
                        . venv/bin/activate
                        python3 model_training.py
                    '''
                    
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        python3 model_evaluation.py
                    '''
                }
            }
        }

        stage('Serve Model') {
            steps {
                script {
                    // Start FastAPI server in the background
                    sh ''' 
                        . venv/bin/activate
                        nohup uvicorn main:app --host
                        '''
                    sleep time: 10, unit: 'SECONDS'
                }
            }
        }

        stage('Test Serve Model') {
            steps {
                script {
                    // Test the server with sample values
                    sh '''
                        curl -X POST "http://127.0.0.1:9000/predict" ^
                        -H "Content-Type: application/json" ^
                        -d "{\\"features\\": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}"
                    '''
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**.pkl', fingerprint: true
            echo 'Pipeline execution complete.'
        }
    }
}