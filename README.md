# Detector de emoções em um texto usando rede neural

<div align="justify">

## Descrição do Projeto:

Este projeto é um exemplo completo de um pipeline de processamento de linguagem natural para a detecção de emoções em textos usando uma rede neural. O objetivo principal é treinar um modelo para classificar textos de acordo com as emoções expressas em seu conteúdo, como alegria, tristeza, raiva, medo, surpresa, entre outras.
Etapas do Projeto:

- ### Leitura dos Dados:
    - Os dados de treinamento, validação e teste são lidos a partir de arquivos de texto. Os dados contêm textos e seus respectivos rótulos, representando diferentes emoções.

- ### Pré-processamento do Texto:
    - O texto é pré-processado para facilitar a manipulação e melhorar a eficiência do modelo. Isso inclui a remoção de caracteres não alfabéticos, conversão do texto para minúsculas, divisão do texto em palavras (tokens), remoção de palavras de parada (stopwords) e aplicação de stemming para reduzir as palavras às suas raízes morfológicas usando PorterStemmer.

- ### Representação do Texto (Bag of Words):
    - O texto pré-processado é convertido em vetores numéricos usando a técnica Bag of Words (BoW). O CountVectorizer é utilizado para criar vetores que representam a contagem ou frequência de cada palavra presente no texto.

- ### Criação e Treinamento de Rede Neural:
    - O modelo é uma rede neural sequencial, construída com a biblioteca Keras. O modelo consiste em camadas densas totalmente conectadas. As funções de ativação usadas são relu nas camadas intermediárias e softmax na camada de saída para lidar com a classificação multiclasse. O modelo é compilado com a função de perda sparse_categorical_crossentropy e otimizador adam.
    - O modelo é treinado com os dados de treinamento para aprender a detectar as emoções.

- ### Avaliação do Modelo:
    - O modelo treinado é avaliado com os dados de teste para verificar sua precisão na detecção das emoções.

- ### Previsão de Emoções:
    - O modelo treinado pode ser usado para prever a emoção expressa em novos textos fornecidos pelo usuário.

- ### Salvamento do Modelo e Ferramentas:
    - O modelo treinado, o CountVectorizer e o codificador de rótulos (LabelEncoder) são salvos em arquivos para uso posterior em aplicações.

## Conceitos Importantes:

  - Bag of Words (BoW): Uma técnica de representação de texto em que cada documento é convertido em um vetor com a contagem ou frequência de cada palavra presente.

  - Stopwords: Palavras comuns em um idioma que são geralmente removidas dos textos para melhorar a eficiência dos modelos, pois não adicionam valor semântico significativo.

  - Stemming: O processo de reduzir palavras a suas formas radicais (raízes) para normalizar as palavras de forma a lidar com variações como plural, singular ou diferentes tempos verbais.

  - Rede Neural: Um modelo de aprendizado de máquina inspirado na estrutura do cérebro humano. Consiste em camadas de neurônios interconectados que processam entradas para gerar saídas.

  - Keras: Uma biblioteca de alto nível para construir redes neurais usando o TensorFlow como backend. Ela facilita o processo de construção, compilação e treinamento de modelos.

  - Sequential: Um tipo de modelo de rede neural em Keras em que as camadas são empilhadas sequencialmente, uma após a outra.

  - Camada Densa: Uma camada de rede neural onde cada neurônio está conectado a todos os neurônios da camada anterior. Processa a entrada com uma função de ativação para gerar a saída.

  - Neurônios: Unidades básicas em uma rede neural que processam as entradas, aplicam uma função de ativação e produzem uma saída.

  - Funções de Ativação: Controlam como o modelo processa a entrada e gera a saída. As principais funções de ativação neste projeto são relu e softmax.

  - Função de Perda: Mede a diferença entre as previsões do modelo e os valores reais. Neste projeto, a função de perda usada é sparse_categorical_crossentropy, adequada para problemas de classificação com múltiplas classes de emoções.

  - Otimizador: Método para ajustar os pesos do modelo durante o treinamento para minimizar a função de perda. O otimizador adam é uma escolha popular por sua eficiência e robustez.

## Aplicações dos Conhecimentos Aprendidos:

  Os conceitos e técnicas utilizados neste projeto podem ser aplicados em várias tarefas de processamento de linguagem natural relacionadas à detecção de emoções, como:

  - Análise de sentimentos em mídias sociais.
  - Avaliação da satisfação de clientes em avaliações de produtos e serviços.
  - Identificação de padrões emocionais em comunicações de clientes para melhorar o suporte ao cliente.
  - Análise de emoções em transcrições de chamadas de suporte ao cliente.

Estes conceitos também podem ser aplicados em outras áreas, como inteligência artificial conversacional, para identificar as emoções dos usuários durante interações com chatbots e assistentes virtuais, permitindo respostas mais empáticas e personalizadas.
