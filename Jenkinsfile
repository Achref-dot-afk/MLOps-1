pipeline {
    agent any

    stages {
        stage('Initialize') {
            steps {
                script {
                    // Install necessary Python packages
                    sh "pip install -r requirements.txt"
                }
            }
        }

        stage('Load and Preprocess Data') {
            steps {
                script {
                    // Run data loading script
                    sh "python data_loading.py"
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    // Run model training script
                    sh "python model_training.py"
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    // Run model evaluation script
                    sh "python model_evaluation.py"
                }
            }
        }

        stage('Serve Model') {
            steps {
                script {
                    // Start FastAPI server in the background
                    sh 'start /B python model_serving.py'
                    // Wait for the server to start
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