# Language Model Evaluation Harness for Portuguese LLMs


**A framework for evaluating Large Language Models (LLMs) in Portuguese**

This repository is a fork of [EleutherAI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), adapted specifically for evaluating language models in Portuguese. 📐 It serves as the evaluation suite for the 🚀 [Open Portuguese LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard), which aims to track, rank, and evaluate open LLMs and chatbots tailored for the Portuguese language.

Submit a model for automated evaluation on the Hugging Face GPU cluster via the ["Submit" page](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?tab=submit) on the leaderboard!

## About the Leaderboard

The 🚀 Open Portuguese LLM Leaderboard aims to provide a comprehensive benchmark for evaluating Large Language Models (LLMs) in the Portuguese language across a variety of tasks and datasets. The leaderboard:

- Is open to submissions from the community
- Serves as a resource for researchers, practitioners, and enthusiasts 
- Includes tasks covering multiple aspects of language understanding and generation

This leaderboard is made possible by the support of the [Center of Excellence in AI (CEIA)](https://ceia.ufg.br/) at the [Federal University of Goiás (UFG)](https://international.ufg.br/).

## Portuguese-Specific Features

This fork includes several modifications tailored for Portuguese language evaluation:

- **Portuguese Task Suite**: A comprehensive collection of NLP tasks designed for the Portuguese language (see list below).
- **Direct Response Evaluation**: Works with models' direct text responses rather than just log probabilities, suitable for evaluating instruction-tuned models and chatbots.
- **Chat Template Support**: Enhanced compatibility with LLMs using various chat templates from the transformers libary. Automatically detects and applies the appropriate chat template format (system-user-assistant, user-assistant, or assistant-user) without requiring manual configuration. This ensures accurate evaluation of chat-optimized models with their native prompt formats.
- **Multi-Backend Support**: 
  - **vLLM Integration**: Accelerated inference with batch evaluation for faster processing of large models.
  - **LiteLLM Support**: Evaluation of closed-source models via APIs (including OpenAI, Google's Vertex AI/Gemini, etc.).
- **Memory Optimization**: 
  - Automatic batch size detection and adjustment based on available GPU memory.
  - Dynamic max_length adjustment for efficient resource utilization.
  - Starting_max_length option for better GPU memory management.
- **Evaluation Enhancements**:
  - Improved metrics calculation (F1-macro, Pearson) with better handling of edge cases.
  - Reasoning extraction for models that provide explanations before answers (e.g., DeepSeek models).
  - Temperature control (set to 0) for API models to ensure deterministic outputs.
- **Custom Filters**: Special text processing pipelines adapted for Portuguese tasks characteristics.
- **UTF-8 Support**: Proper encoding for Portuguese text with accents and special characters in both inputs and outputs.
- **Few-shot ID Sampling**: Preserves order of few-shot examples for consistent evaluation.

## Portuguese Evaluation Tasks

The evaluation suite includes a diverse set of tasks covering different capabilities. The evaluations primarily use few-shot examples (typically 3 to 25, depending on the task) to assess model performance in context.

| Task Alias      | Description                                           | Few-shot | Main Metric | Baseline | Link/Source                                                                 |
|-----------------|-------------------------------------------------------|----------|-------------|----------|-----------------------------------------------------------------------------|
| **assin2_rte**  | Recognizing Textual Entailment (ASSIN 2)              | 15       | F1 Macro    | 50.0     | [ASSIN 2](https://sites.google.com/view/assin2/)                            |
| **assin2_sts**  | Semantic Textual Similarity (ASSIN 2)                 | 15       | Pearson     | 0.0      | [ASSIN 2](https://sites.google.com/view/assin2/)                            |
| **bluex**       | Reading Comprehension (BlueX)                         | 5        | F1 Macro    | 33.3     | [BlueX Dataset](https://github.com/dlicari/bluex)                           |
| **enem**        | Multiple Choice Questions (ENEM Exam)                 | 3        | Accuracy    | 20.0     | [ENEM Challenge](https://www.ime.usp.br/~ddm/project/enem/) |
| **faquad_nli**  | Natural Language Inference (FaQuAD-NLI)               | 5        | F1 Macro    | 33.3     | [FaQuAD-NLI](https://huggingface.co/datasets/rubensms/FaQuAD-NLI)           |
| **hatebr**      | Offensive Language Detection (HateBR)                 | 25       | F1 Macro    | 50.0     | [HateBR Dataset](https://github.com/romeropeixoto/HateBR)                   |
| **hate_speech** | Hate Speech Identification (Portuguese Hate Speech)   | 25       | F1 Macro    | 47.9     | [Portuguese Hate Speech](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset) |
| **tweetsentbr** | Sentiment Analysis (TweetSentBR)                      | 25       | F1 Macro    | 32.8     | [TweetSentBR](https://bitbucket.org/HBrum/tweetsentbr)                      |
| **oab_exams**   | Brazilian Bar Exam Questions                          | 3        | Accuracy    | 20.0     | [OAB Exams](https://github.com/legal-nlp/oab-exams)           |

Task descriptions:

- **assin2_rte**: A dataset for Recognizing Textual Entailment in Portuguese, part of the ASSIN 2 shared task.
- **assin2_sts**: A dataset for Semantic Textual Similarity in Portuguese, assessing model ability to determine semantic equivalence between sentences.
- **bluex**: A reading comprehension dataset for Portuguese, testing the ability to understand and extract information from texts.
- **enem**: Questions from the Brazilian High School National Exam (ENEM), covering various subjects in multiple-choice format.
- **faquad_nli**: A Natural Language Inference dataset derived from the FaQuAD question-answering dataset for Portuguese.
- **hatebr**: A dataset of Brazilian Instagram comments annotated for offensive language and hate speech detection.
- **hate_speech**: A hierarchically labeled Portuguese hate speech dataset composed of tweets with binary annotations.
- **tweetsentbr**: A corpus of tweets in Brazilian Portuguese annotated for sentiment analysis in three classes (Positive, Negative, Neutral).
- **oab_exams**: Multiple-choice questions from the Brazilian Bar Exam, testing legal knowledge and reasoning.

*Note: Baseline scores represent the default performance expectation (e.g., random guessing for classification tasks). Few-shot counts might vary slightly based on configuration.*

## Getting Started

### Installation

```bash
git clone https://github.com/eduagarcia/lm-evaluation-harness-pt
cd lm-evaluation-harness-pt
pip install -e .

# For extended functionality (faster inference with vLLM, API access, etc.)
pip install -e ".[vllm,anthropic,openai,sentencepiece]"
```

### Basic Usage

To evaluate a Portuguese LLM with the complete Open PT LLM Leaderboard benchmark:

```bash
lm_eval \
    --model huggingface \
    --model_args "pretrained=YOUR_MODEL_ID,revision=main" \
    --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
    --device cuda:0 \
    --output_path "./"
```

You can also evaluate individual tasks:

```bash
# For base models
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL_ID,trust_remote_code=True \
    --tasks assin2_rte,tweetsentbr \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/YOUR_MODEL_ID
```

Chat Template - The libary automatically detects and applies chat templates when they exist in the model's tokenizer config. If you need to disable chat template:

```bash
# It tests different chat formats (system-user-assistant, user-assistant, assistant-user) 
# and uses the one that works with your model.
# Disable chat template detection
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL_ID,trust_remote_code=True,apply_chat_template=False \
    --tasks assin2_rte,tweetsentbr \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/YOUR_MODEL_ID
```

*Set `batch_size` to `auto` for automatic batch size detection or specify an integer value.*

### Memory Optimization

Choose the optimization technique that best fits your hardware constraints:

#### Automatic Batch Size
```bash
# Auto-detect the largest possible batch size for your GPU
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL_ID \
    --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/YOUR_MODEL_ID
```


#### Starting Max Length
```bash
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL_ID,starting_max_length=1024 \
    --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/YOUR_MODEL_ID
```

#### 4-bit Quantization
```bash
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL_ID,load_in_4bit=True \
    --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/YOUR_MODEL_ID
```

### Using API-based Models

For evaluating proprietary models through APIs (e.g., OpenAI):

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-chat-completions \
    --model_args model=gpt-4-turbo \
    --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
    --output_path results/gpt-4-turbo
```

## Submitting to the Leaderboard Manually

The [Open Portuguese LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) offers two methods for submitting models for evaluation:

1. **Automatic Submission**: Submit your model through the leaderboard's ["Submit" page](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?tab=submit) for evaluation on a available GPU cluster. This is the recommended and simplest approach for most models.

2. **Manual Submission**: For models with special requirements (such as those requiring `trust_remote_code=True`, depending on external libraries beyond transformers, or models without publicly accessible weights), or when automatic submission encounters issues. Detailed instructions for manual submission are provided below.

For manual submission, please follow these steps:

### 1. Run the Evaluation

Execute the evaluation harness for the complete benchmark:
```bash
lm_eval --model hf \
       --model_args pretrained=YOUR_MODEL_ID,trust_remote_code=True \
       --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr \
       --device cuda:0 \
       --batch_size auto \
       --output_path results/YOUR_MODEL_ID \
       --log_samples
```

### 2. Submit Results

After running the evaluation:
1. Look for the results JSON file in your output directory
2. [Open an issue](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard/discussions/new) on the leaderboard space with your results attached
3. **Send an email** to edusantosgarcia@gmail.com with:
   - Your model name and Hugging Face Hub link
   - The zipped JSON files attached
   - Any special considerations about your model

I will review and add your results manually.

## Troubleshooting Evaluation Failures

If your model evaluation fails:
1  **Local Test:** Try running the evaluation command locally first. You can add `--limit 10` to quickly test evaluation on a small number of examples per task.
    ```bash
    lm_eval --model hf \
        --model_args pretrained=YOUR_MODEL_ID,trust_remote_code=True \
        --tasks pt_benchmark \
        --device cuda:0 \
        --batch_size 8 \
        --limit 10
    ```
2.  **Check Logs:** Examine the output logs for specific error messages.
3.  **Consult Leaderboard FAQ:** Review the FAQ section on the [leaderboard space](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) for common issues.
4.  **Open Issue:** If the problem persists, consider opening an issue in this repository or on the leaderboard's discussion forum.

## Acknowledgments

This project builds upon the excellent work of [EleutherAI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). We express our gratitude to the original authors and contributors. We also thank the creators of the datasets used in the Portuguese benchmark tasks.

## Citation

If you use this framework or the benchmark results in your research, please cite this repository and the original LM Evaluation Harness. Consider citing the specific datasets used as well.

```bibtex
@misc{open-pt-llm-leaderboard,
  author = {Garcia, Eduardo A. S.},
  title = {Open Portuguese LLM Leaderboard},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = "\url{https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard}"
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = {2023},
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```

For specific datasets used in the Portuguese benchmark:

```bibtex
@InProceedings{ENEM-Challenge,
  author = {Silveira, Igor Cataneo and Mau\'a, Denis Deratani},
  booktitle = {Proceedings of the 6th Brazilian Conference on Intelligent Systems},
  series = {BRACIS},
  title = {University Entrance Exam as a Guiding Test for Artificial Intelligence},
  pages = {426--431},
  year = {2017}
}

@misc{nunes2023evaluating,
  title={Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams}, 
  author={Desnes Nunes and Ricardo Primi and Ramon Pires and Roberto Lotufo and Rodrigo Nogueira},
  year={2023},
  eprint={2303.17003},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{pires2023evaluating,
  title={Evaluating GPT-4's Vision Capabilities on Brazilian University Admission Exams}, 
  author={Ramon Pires and Thales Sales Almeida and Hugo Abonizio and Rodrigo Nogueira},
  year={2023},
  eprint={2311.14169},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{almeida2023bluex,
  title={BLUEX: A benchmark based on Brazilian Leading Universities Entrance eXams}, 
  author={Thales Sales Almeida and Thiago Laitz and Giovana K. Bonás and Rodrigo Nogueira},
  year={2023},
  eprint={2307.05410},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{d2017passing,
  title={Passing the Brazilian OAB Exam: Data Preparation and Some Experiments1},
  author={d RADEMAKER, Alexandre},
  booktitle={Legal Knowledge and Information Systems: JURIX 2017: The Thirtieth Annual Conference},
  volume={302},
  pages={89},
  year={2017},
  organization={IOS Press}
}

@inproceedings{real2020assin,
  title={The assin 2 shared task: a quick overview},
  author={Real, Livy and Fonseca, Erick and Oliveira, Hugo Goncalo},
  booktitle={International Conference on Computational Processing of the Portuguese Language},
  pages={406--412},
  year={2020},
  organization={Springer}
}

@inproceedings{8923668,
  author={Sayama, Hélio Fonseca and Araujo, Anderson Viçoso and Fernandes, Eraldo Rezende},
  booktitle={2019 8th Brazilian Conference on Intelligent Systems (BRACIS)}, 
  title={FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education}, 
  year={2019},
  volume={},
  number={},
  pages={443-448},
  keywords={Training;Context modeling;Encyclopedias;Electronic publishing;Internet;Natural Language Processing;Machine Reading Comprehension;Dataset},
  doi={10.1109/BRACIS.2019.00084}
}

@software{Chaves_Rodrigues_napolab_2023,
  author = {Chaves Rodrigues, Ruan and Tanti, Marc and Agerri, Rodrigo},
  doi = {10.5281/zenodo.7781848},
  month = {3},
  title = {{Natural Portuguese Language Benchmark (Napolab)}},
  url = {https://github.com/ruanchaves/napolab},
  version = {1.0.0},
  year = {2023}
}

@inproceedings{vargas-etal-2022-hatebr,
  title = "{H}ate{BR}: A Large Expert Annotated Corpus of {B}razilian {I}nstagram Comments for Offensive Language and Hate Speech Detection",
  author = "Vargas, Francielle  and
    Carvalho, Isabelle  and
    Rodrigues de G{\'o}es, Fabiana  and
    Pardo, Thiago  and
    Benevenuto, Fabr{\'\i}cio",
  booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
  month = jun,
  year = "2022",
  address = "Marseille, France",
  publisher = "European Language Resources Association",
  url = "https://aclanthology.org/2022.lrec-1.777",
  pages = "7174--7183"
}

@inproceedings{fortuna-etal-2019-hierarchically,
  title = "A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset",
  author = "Fortuna, Paula  and
    Rocha da Silva, Jo{\~a}o  and
    Soler-Company, Juan  and
    Wanner, Leo  and
    Nunes, S{\'e}rgio",
  booktitle = "Proceedings of the 3rd Workshop on Abusive Language Online (ALW3)",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/W19-3510",
  doi = "10.18653/v1/W19-3510",
  pages = "94--104",
}

@InProceedings{BRUM18.389,
  author = {Henrico Brum and Maria das Gra\c{c}as Volpe Nunes},
  title = "{Building a Sentiment Corpus of Tweets in Brazilian Portuguese}",
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {May 7-12, 2018},
  address = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and HÚlŔne Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {979-10-95546-00-9},
  language = {english}
}
```