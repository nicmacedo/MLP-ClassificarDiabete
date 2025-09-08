# 🧠 Rede Neural MLP para Classificação de Diabetes

Este projeto implementa uma Rede Neural Multilayer Perceptron (MLP) para classificação de pacientes com diabetes baseada em Índice de Massa Corporal (IMC) e nível de Glicemia.

## 📋 Descrição

O sistema utiliza uma arquitetura neural com:
- **Camada de entrada**: 2 neurônios (IMC + Glicemia) + bias
- **Camada oculta**: 3 neurônios com função de ativação sigmóide
- **Camada de saída**: 2 neurônios com função linear para classificação binária

## 🎯 Objetivo

Classificar pacientes em duas categorias:
- 🔴 **Com diabetes** (saída: [1, 0])
- 🟢 **Sem diabetes** (saída: [0, 1])

## 📊 Dataset

O conjunto de dados sintético contém 16 exemplos com:

| Atributo | Faixa | Normalização |
|----------|-------|-------------|
| IMC | 16-40 kg/m² | `(valor - 16) / (40 - 16)` |
| Glicemia | 70-126 mg/dL | `(valor - 70) / (126 - 70)` |

**Distribuição:**
- 📚 10 exemplos para treinamento
- 🧪 6 exemplos para teste/validação

## 🏗️ Arquitetura da Rede

```
Input (2+1) → Hidden Layer (3 neurônios) → Output Layer (2 neurônios)
      ↑            ↑                             ↑
     IMC        Sigmoid                        Linear
   Glicemia      Bias                           Bias
```

## ⚙️ Funcionalidades

- ✅ Normalização automática dos dados
- ✅ Separação treino/teste
- ✅ Retropropagação com taxa de aprendizado ajustável
- ✅ Aleatorização dos dados por época
- ✅ Métricas de avaliação de desempenho
- ✅ Detecção de overfitting

## 🚀 Como Executar

```bash
# Compilar
g++ -o mlp_diabetes mlp_diabetes.cpp -std=c++11

# Executar
./mlp_diabetes
```

## 📈 Resultados Esperados

O programa exibirá:
- Progresso do treinamento a cada 100 épocas
- Resultados com dados de treino (10 exemplos)
- Resultados com dados de teste (6 exemplos)
- Acurácia em ambos os conjuntos
- Diagnóstico de overfitting

## 📝 Exemplo de Saída

```
=== RESULTADOS COM DADOS DE TESTE ===
Paciente 11: IMC=25, Glic=70 -> SEM DIABETES [0.12, 0.85] Correto
Paciente 12: IMC=20, Glic=72 -> SEM DIABETES [0.08, 0.91] Correto
...

=== DESEMPENHO DA REDE ===
Acurácia no treino: 100.00%
Acurácia no teste: 83.33%
Rede generalizando bem!
```

## 🛠️ Personalização

É possível ajustar os parâmetros no código:

```cpp
int epocas = 1000;                   // Número de épocas de treinamento
float taxaAprendizado = 0.1f;        // Taxa de aprendizado
```

## 📚 Referências

- algoritmo de Backpropagation
- Função de ativação Sigmóide: `φ(v) = 1 / (1 + e^(-v))`
- Normalização min-max para pré-processamento
- Técnicas de prevenção de overfitting

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar issues
- Sugerir melhorias
- Enviar pull requests

## 📄 Licença

Este projeto está sob a licença MIT.

---

**Nota**: Este é um projeto educacional com dados sintéticos. Não use para diagnósticos médicos reais.
