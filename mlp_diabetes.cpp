#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>
using namespace std;

// FunC'C#o da normalizaC'C#o dos dados entre 0 e 1
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

	// Camada de SaC-da 2 neuronios
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
		{1, 16, 75, 1, 0},
		{1, 25, 110, 1, 0},
		{1, 38, 126, 1, 0},
		{1, 25, 70, 0, 1},
		{1, 20, 72, 0, 1},
		{1, 35, 99, 0, 1},
		{1, 32, 120, 1, 0},
		{1, 31, 110, 1, 0},
		{1, 33, 95, 0, 1}
	};

	// NormalizaC'C#o dos dados entre 0 e 1
	float dadosNorm[16][5];
	normalizarDados(dados, dadosNorm);

	// InicializaC'C#o dos pesos
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

	// ConfiguraC'C#o para aleatorizaC'C#o
	random_device rd;
	mt19937 g(rd());

	// Separar dados: 10 para treino, 6 para teste
	vector<int> indices_treino = {0, 1, 2, 3, 4, 5, 6, 8, 13, 15};
	vector<int> indices_teste = {10, 11, 12, 7, 14, 9};

	// Treinamento da rede neural apenas com dados de treino
	for(int e = 0; e < epocas; e++) {
		// Aleatorizar a ordem dos dados de treino a cada C)poca
		shuffle(indices_treino.begin(), indices_treino.end(), g);

		for(int idx = 0; idx < indices_treino.size(); idx++) {
			int linha = indices_treino[idx];

			// Reset das camadas
			for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
			for(int i = 0; i < 2; i++) camadaSaida[i] = 0;

			// PropagaC'C#o - Camada Oculta
			for(int i = 0; i < 3; i++) {
				for(int j = 0; j < 3; j++) {
					camadaOculta[i] += dadosNorm[linha][j] * pesosCamadaOculta[i][j];
				}
				camadaOculta[i] = 1 / (1 + exp(-camadaOculta[i]));
			}

			// PropagaC'C#o - Camada SaC-da
			for(int i = 0; i < 2; i++) {
				camadaSaida[i] = pesosCamadaSaida[i][0];
				for(int j = 1; j < 4; j++) {
					camadaSaida[i] += camadaOculta[j-1] * pesosCamadaSaida[i][j];
				}
			}

			// RetropropagaC'C#o - Camada SaC-da
			float deltaSaida[2];
			for(int i = 0; i < 2; i++) {
				float erro = dadosNorm[linha][i+3] - camadaSaida[i];
				deltaSaida[i] = erro * 1;
			}

			// RetropropagaC'C#o - Camada Oculta
			float deltaOculta[3];
			for(int i = 0; i < 3; i++) {
				float soma = 0;
				for(int j = 0; j < 2; j++) {
					soma += deltaSaida[j] * pesosCamadaSaida[j][i+1];
				}
				deltaOculta[i] = camadaOculta[i] * (1 - camadaOculta[i]) * soma;
			}

			// AtualizaC'C#o dos pesos - Camada SaC-da
			for(int i = 0; i < 2; i++) {
				pesosCamadaSaida[i][0] += taxaAprendizado * deltaSaida[i] * 1; // Bias
				for(int j = 1; j < 4; j++) {
					pesosCamadaSaida[i][j] += taxaAprendizado * deltaSaida[i] * camadaOculta[j-1];
				}
			}

			// AtualizaC'C#o dos pesos - Camada Oculta
			for(int i = 0; i < 3; i++) {
				pesosCamadaOculta[i][0] += taxaAprendizado * deltaOculta[i] * 1; // Bias
				for(int j = 1; j < 3; j++) {
					pesosCamadaOculta[i][j] += taxaAprendizado * deltaOculta[i] * dadosNorm[linha][j];
				}
			}
		}

		// Exibir progresso a cada 100 C)pocas
		if(e % 100 == 0) {
			cout << "Epoca: " << e << " concluida" << endl;
		}
	}

	// Teste da rede treinada com dados de TESTE
	cout << "\n=== RESULTADOS COM DADOS DE TREINO ===" << endl;
	int acertos_treino = 0 ,fn_treino = 0, tp_treino = 0, fp_treino = 0;
	float precisao_treino = 0, recall_treino = 0, f1_score_treino = 0;
	for(int idx = 0; idx < indices_treino.size(); idx++) {
		int linha = indices_treino[idx];

		// Reset das camadas
		for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
		for(int i = 0; i < 2; i++) camadaSaida[i] = 0;

		// Propagacao para teste
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

		// Normalização das saídas
        if(camadaSaida[0] >= 0.5) camadaSaida[0] = 1;
        else camadaSaida[0] = 0;
        
        if(camadaSaida[1] >= 0.5) camadaSaida[1] = 1;
        else camadaSaida[1] = 0;
        
       // Determinar a classe prevista

		string classe_real;
		if(dados[linha][3] == 1) {
			classe_real = "COM DIABETES";
		} else {
			classe_real = "SEM DIABETES";
		}

		string classif;
		string acerto = (camadaSaida[0] == dados[linha][3] && camadaSaida[1] == dados[linha][4]) ? "Correto" : "Errado";
		if(acerto == "Errado") classif = (classe_real == "COM DIABETES") ? "FN" : "TN";
		else classif = (classe_real == "COM DIABETES") ? "TP" : "TN";

        //calculo para precisao, recall e f1-score
		if(classif == "FN") fn_treino++;
		if(classif == "TP") tp_treino++;
		if(classif == "FP") fp_treino++;

		if (acerto == "Correto") acertos_treino++;

		cout << "Paciente " << linha+1 << ": IMC=" << dados[linha][1]
		     << ", Glic=" << dados[linha][2] << " | "
		     << "Saída desejada" << " -> " << "[" << dados[linha][3] <<  ", " << dados[linha][4] << "]" << " | " 
		     << "Saída da rede" << " -> " <<" [" << fixed << setprecision(0) << camadaSaida[0] << ", " << camadaSaida[1] << "] "
		     << acerto << "(" << classif << ")" << endl;
	}
	cout << "Acertos no treino: " << acertos_treino << "/10" << endl;

	// Teste da rede treinada com dados de TESTE
	cout << "\n=== RESULTADOS COM DADOS DE TESTE ===" << endl;
	int acertos_teste = 0, fn_teste = 0, tp_teste = 0, fp_teste = 0;
	float precisao_teste = 0, recall_teste = 0, f1_score_teste = 0;
	
	for(int idx = 0; idx < indices_teste.size(); idx++) {
		int linha = indices_teste[idx];

		// Reset das camadas
		for(int i = 0; i < 3; i++) camadaOculta[i] = 0;
		for(int i = 0; i < 2; i++) camadaSaida[i] = 0;

		// PropagaC'C#o para teste
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

		// Normalização das saídas
        if(camadaSaida[0] >= 0.5) camadaSaida[0] = 1;
        else camadaSaida[0] = 0;
        
        if(camadaSaida[1] >= 0.5) camadaSaida[1] = 1;
        else camadaSaida[1] = 0;
        
       // Determinar a classe prevista

		string classe_real;
		if(dados[linha][3] == 1) {
			classe_real = "COM DIABETES";
		} else {
			classe_real = "SEM DIABETES";
		}

		string classif;
		string acerto = (camadaSaida[0] == dados[linha][3] && camadaSaida[1] == dados[linha][4]) ? "Correto" : "Errado";
		if(acerto == "Errado") classif = (classe_real == "COM DIABETES") ? "FN" : "TN";
		else classif = (classe_real == "COM DIABETES") ? "TP" : "TN";

        //calculo para precisao, recall e f1-score
		if(classif == "FN") fn_teste++;
		if(classif == "TP") tp_teste++;
		if(classif == "FP") fp_teste++;

		if (acerto == "Correto") acertos_teste++;
		
		cout << "Paciente " << linha+1 << ": IMC=" << dados[linha][1]
		     << ", Glic=" << dados[linha][2] << " | "
		     << "Saída desejada" << " -> " << "[" << dados[linha][3] <<  ", " << dados[linha][4] << "]" << " | " 
		     << "Saída da rede" << " -> " <<" [" << fixed << setprecision(0) << camadaSaida[0] << ", " << camadaSaida[1] << "] "
		     << acerto << "(" << classif << ")" << endl;
	}
	cout << "Acertos no teste: " << acertos_teste << "/6" << endl;

	// Calcular e exibir a acurC!cia
	float accuracy_treino = (float)acertos_treino / 10 * 100;
	float accuracy_teste = (float)acertos_teste / 6 * 100;

	cout << "\n=== ACURACIA DA REDE ===" << endl;
	cout << "Acuracia no treino: " << fixed << setprecision(2) << accuracy_treino << "%" << endl;
	cout << "Acuracia no teste: " << fixed << setprecision(2) << accuracy_teste << "%" << endl;

	if (accuracy_teste > 80) {
		cout << "Rede generalizando bem!" << endl;
	} else if (accuracy_teste > 60) {
		cout << "Rede com desempenho moderado." << endl;
	} else {
		cout << "Rede com overfitting (apenas memorizou os dados de treino)." << endl;
	}
	
	//Calcular e exibir a precisão
	precisao_treino = (float)tp_treino / (tp_treino + fp_treino);
	precisao_teste = (float)tp_teste / (tp_teste + fp_teste);
	
	cout << "\n===PRECISAO DA REDE ===" << endl;
	cout << "Precisao no treino: " << fixed << setprecision(2) << precisao_treino << endl;
	cout << "Precisao no teste: " << fixed << setprecision(2) << precisao_teste << endl;

    //Calcular e exibir o recall
    recall_treino = (float)tp_treino / (tp_treino + fn_treino);
    recall_teste = (float)tp_teste / (tp_teste + fn_teste);
    
    cout << "\n===RECALL DA REDE ===" << endl;
	cout << "RECALL no treino: " << fixed << setprecision(2) << recall_treino << endl;
	cout << "RECALL no teste: " << fixed << setprecision(2) << recall_teste << endl;

    //Calcular e exibir o F1-score
    f1_score_treino = (float) 2*(precisao_treino * recall_treino)/(precisao_treino + recall_treino);
    f1_score_teste = (float) 2*(precisao_teste * recall_teste)/(precisao_teste +recall_teste);
    
    cout << "\n===F1-SCORE DA REDE ===" << endl;
	cout << "F1-score no treino: " << fixed << setprecision(2) << f1_score_treino << endl;
	cout << "F1-score no teste: " << fixed << setprecision(2) << f1_score_teste << endl;
    
	return 0;
}
