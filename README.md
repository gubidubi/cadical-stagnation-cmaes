# Otimização de Hiperparâmetros do CaDiCaL com Restart por Estagnação

> ⚠️ **Nota sobre o estado do repositório:**  
> Este repositório consiste em uma **cópia direta (*snapshot*) dos arquivos na máquina local** utilizados durante o Trabalho de Graduação (Engenharia de Computação – ITA). O conteúdo **não foi extensivamente reorganizado, limpo ou refatorado** para facilitar o uso geral ou distribuição como pacote de software. Trata-se de um arquivo bruto disponibilizado para transparência acadêmica e preservação do histórico experimental.

Os *scripts* foram desenvolvidos originalmente para o ambiente computacional do Cluster Euler (USP/CeMEAI) e contêm referências a caminhos de diretórios e variáveis específicas. Para qualquer tentativa de reprodução, é necessário consultar o arquivo `euler` e replicar a árvore de pastas esperada.

---

## Estrutura de Arquivos e Pastas

* **`Benchmarks/`**: Arquivos e dados de saída dos ensaios comparativos de benchmark.
* **`CMAES/`**: *Scripts*, logs de execução e arquivos de saída relativos ao processo de otimização estocástica via CMA-ES.
* **`lists/`**: Listas em formato texto contendo os caminhos e referências das instâncias CNF utilizadas pelos *scripts*.
* **`Resultados/`**: Rotinas de pós-processamento, tabelas consolidadas e dados de geração dos gráficos da monografia.
* **`Scripts/`**: Coleção de rotinas em Python e scripts de automação/submissão utilizados na otimização e validação.
* **`dificuldades`**: Registro informal em texto com notas operacionais e problemas encontrados ao longo da execução dos experimentos.
* **`euler`**: Documentação técnica contendo informações sobre a estrutura do ambiente remoto no Cluster Euler, hierarquia de diretórios e comandos de compilação.
* **`requirements.txt`**: Lista de dependências Python para reconstrução do ambiente virtual (`meu_env`).
* **`README.md`**: Este documento de contextualização.

---

## Ambiente e Execução

Para reproduzir os ensaios ou navegar pelos scripts:

1. **Dependência de caminhos:** Consulte o arquivo `euler` para compreender a árvore de diretórios na qual os scripts foram executados. Muitos caminhos estão configurados de forma absoluta ou dependem da hierarquia original do cluster.
2. **Ambiente Python:** Instale as bibliotecas registradas no `requirements.txt`:
   ```bash
   python3 -m venv meu_env
   source meu_env/bin/activate
   pip install -r requirements.txt
