#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>
using namespace std;

// Função da normalização dos dados entre 0 e 1
void normalizarDados(float dados[16][5], float dadosNorm[16][5]) {
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 5; j++) {
            if(j == 1) {
                dadosNorm[i][j] = (dados[i][j] - 16) / (40 - 16);
            }
            else if(j == 2) {
                dadosNorm[i][j] = (dados[i][j] - 70) / (126 - 70);
            }
            else {
                dadosNorm[i][j] = dados[i][j];
            }
        }
    }
}

int main() {
    int epocas = 1000;
    float taxaAprendizado = 0.1f;
    
    // Camada Oculta 3 neuronios
    float pesosCamadaOculta[3][3];
    float camadaOculta[3] = {0, 0, 0};
    
    // Camada de Saída 2 neuronios
    float pesosCamadaSaida[2][4];
    float camadaSaida[2] = {0, 0};
    
    float dados[16][5] = {
        //bias, imc, glicemia, saida 1(com diabetes), saida 2(sem diabetes)
        {1, 16, 70, 0, 1},
        {1, 18, 80, 0, 1},
        {1, 20, 90, 0, 1},
        {1, 40, 126, 1, 0},
        {1, 30, 125, 1, 0},
        {1, 22, 100, 0, 1},
        {1, 35, 122, 1, 0},
        {1, 16, 75, 0, 1},
        {1, 25, 110, 0, 1},
        {1, 38, 126, 1, 0},
        {1, 25, 70, 0, 1},
        {1, 20, 72, 0, 1},
        {1, 35, 99, 1, 0},
        {1, 32, 120, 0, 1},
        {1, 31, 110, 0, 1},
        {1, 33, 95, 0, 1}
    };

    // Normalização dos dados entre 0 e 1
    float dadosNorm[16][5];
    normalizarDados(dados, dadosNorm);

    // Inicialização dos pesos
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            pesosCamadaOculta[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }  

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            pesosCamadaSaida[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }

    // Configuração para aleatorização
    random_device rd;
    mt19937 g(rd());
    
    // Separar dados: 10 para treino, 6 para teste
    vector<int> indices_treino = {0, 1, 2, 3, 4, 5, 6, 8, 13, 15};
    vector<int> indices_teste = {10, 11, 12, 7, 14, 9};         

    // Treinamento da rede neural apenas com dados de treino
    for(int e = 0; e < epocas; e++) {
        // Aleatorizar a ordem dos dados de treino a cada época
        shuffle(indices_treino.begin(), indices_treino.end(), g);

        for(int idx = 0; idx < indices_treino.size(); idx++) {
            int linha = indices_treino[idx];
            
            // Reset das camadas
            for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
            for(int i = 0; i < 2; i++) camadaSaida[i] = 0;
            
            // Propagação - Camada Oculta
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    camadaOculta[i] += dadosNorm[linha][j] * pesosCamadaOculta[i][j];
                }
                camadaOculta[i] = 1 / (1 + exp(-camadaOculta[i]));
            }
            
            // Propagação - Camada Saída
            for(int i = 0; i < 2; i++) {
                camadaSaida[i] = pesosCamadaSaida[i][0]; 
                for(int j = 1; j < 4; j++) {
                    camadaSaida[i] += camadaOculta[j-1] * pesosCamadaSaida[i][j];
                }
            }
            
            // Retropropagação - Camada Saída
            float deltaSaida[2];
            for(int i = 0; i < 2; i++) {
                float erro = dadosNorm[linha][i+3] - camadaSaida[i];
                deltaSaida[i] = erro * 1; 
            }
            
            // Retropropagação - Camada Oculta
            float deltaOculta[3];
            for(int i = 0; i < 3; i++) {
                float soma = 0;
                for(int j = 0; j < 2; j++) {
                    soma += deltaSaida[j] * pesosCamadaSaida[j][i+1];
                }
                deltaOculta[i] = camadaOculta[i] * (1 - camadaOculta[i]) * soma;
            }
            
            // Atualização dos pesos - Camada Saída
            for(int i = 0; i < 2; i++) {
                pesosCamadaSaida[i][0] += taxaAprendizado * deltaSaida[i] * 1; // Bias
                for(int j = 1; j < 4; j++) {
                    pesosCamadaSaida[i][j] += taxaAprendizado * deltaSaida[i] * camadaOculta[j-1];
                }
            }
            
            // Atualização dos pesos - Camada Oculta
            for(int i = 0; i < 3; i++) {
                pesosCamadaOculta[i][0] += taxaAprendizado * deltaOculta[i] * 1; // Bias
                for(int j = 1; j < 3; j++) {
                    pesosCamadaOculta[i][j] += taxaAprendizado * deltaOculta[i] * dadosNorm[linha][j];
                }
            }
        }
        
        // Exibir progresso a cada 100 épocas
        if(e % 100 == 0) {
            cout << "Epoca: " << e << " concluida" << endl;
        }
    }

    // Teste da rede treinada com dados de TESTE
    cout << "\n=== RESULTADOS COM DADOS DE TREINO ===" << endl;
    int acertos_treino = 0;
    for(int idx = 0; idx < indices_treino.size(); idx++) {
        int linha = indices_treino[idx];
        
        // Reset das camadas
        for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
        for(int i = 0; i < 2; i++) camadaSaida[i] = 0;

        // Propagação para teste
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                camadaOculta[i] += dadosNorm[linha][j] * pesosCamadaOculta[i][j];
            }
            camadaOculta[i] = 1 / (1 + exp(-camadaOculta[i]));
        }
        
        for(int i = 0; i < 2; i++) {
            camadaSaida[i] = pesosCamadaSaida[i][0];
            for(int j = 1; j < 4; j++) {
                camadaSaida[i] += camadaOculta[j-1] * pesosCamadaSaida[i][j];
            }
        }
        
        // Determinar a classe prevista
        string classe_prevista;
        if(camadaSaida[0] > camadaSaida[1]) {
            classe_prevista = "COM DIABETES";
        } else {
            classe_prevista = "SEM DIABETES";
        }
        
        string classe_real;
        if(dados[linha][3] == 1) {
            classe_real = "COM DIABETES";
        } else {
            classe_real = "SEM DIABETES";
        }
        
        string acerto = (classe_prevista == classe_real) ? "Correto" : "Errado";
        if (classe_prevista == classe_real) acertos_treino++;
        
        cout << "Paciente " << linha+1 << ": IMC=" << dados[linha][1] 
             << ", Glic=" << dados[linha][2] 
             << " -> " << classe_prevista 
             << " [" << fixed << setprecision(3) << camadaSaida[0] << ", " << camadaSaida[1] << "] "
             << acerto << endl;
    }
    cout << "Acertos no treino: " << acertos_treino << "/10" << endl;

    // Teste da rede treinada com dados de TESTE
    cout << "\n=== RESULTADOS COM DADOS DE TESTE ===" << endl;
    int acertos_teste = 0;
    for(int idx = 0; idx < indices_teste.size(); idx++) {
        int linha = indices_teste[idx];
        
        // Reset das camadas
        for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
        for(int i = 0; i < 2; i++) camadaSaida[i] = 0;

        // Propagação para teste
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                camadaOculta[i] += dadosNorm[linha][j] * pesosCamadaOculta[i][j];
            }
            camadaOculta[i] = 1 / (1 + exp(-camadaOculta[i]));
        }
        
        for(int i = 0; i < 2; i++) {
            camadaSaida[i] = pesosCamadaSaida[i][0];
            for(int j = 1; j < 4; j++) {
                camadaSaida[i] += camadaOculta[j-1] * pesosCamadaSaida[i][j];
            }
        }
        
        // Determinar a classe prevista
        string classe_prevista;
        if(camadaSaida[0] > camadaSaida[1]) {
            classe_prevista = "COM DIABETES";
        } else {
            classe_prevista = "SEM DIABETES";
        }
        
        string classe_real;
        if(dados[linha][3] == 1) {
            classe_real = "COM DIABETES";
        } else {
            classe_real = "SEM DIABETES";
        }
        
        string acerto = (classe_prevista == classe_real) ? "Correto" : "Errado";
        if (classe_prevista == classe_real) acertos_teste++;
        
        cout << "Paciente " << linha+1 << ": IMC=" << dados[linha][1] 
             << ", Glic=" << dados[linha][2] 
             << " -> " << classe_prevista 
             << " [" << fixed << setprecision(3) << camadaSaida[0] << ", " << camadaSaida[1] << "] "
             << acerto << endl;
    }
    cout << "Acertos no teste: " << acertos_teste << "/6" << endl;

    // Calcular e exibir a acurácia
    float accuracy_treino = (float)acertos_treino / 10 * 100;
    float accuracy_teste = (float)acertos_teste / 6 * 100;
    
    cout << "\n=== DESEMPENHO DA REDE ===" << endl;
    cout << "Acuracia no treino: " << fixed << setprecision(2) << accuracy_treino << "%" << endl;
    cout << "Acuracia no teste: " << fixed << setprecision(2) << accuracy_teste << "%" << endl;
    
    if (accuracy_teste > 80) {
        cout << "Rede generalizando bem!" << endl;
    } else if (accuracy_teste > 60) {
        cout << "Rede com desempenho moderado." << endl;
    } else {
        cout << "Rede com overfitting (apenas memorizou os dados de treino)." << endl;
    }

    return 0;
}
