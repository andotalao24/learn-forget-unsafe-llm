

# Aprendizaje y Olvido de Ejemplos No Seguros en Grandes Modelos de Lenguaje
Este repositorio contiene el código que acompaña al artículo [Learning and Forgetting Unsafe Examples in Large Language Models](https://arxiv.org/abs/2312.12736). La base de código se construye alrededor del Trainer de Hugging Face Transformers.  


## Estructura  

|-- src/  
||-- postprocess.py \# filtrado de ejemplos no seguros   
||-- process_bbq.py \# preprocesamiento de datos  
||-- process_harmfulqa.py \# preprocesamiento de datos  
||-- process_toxic.py \# preprocesamiento de datos  
||-- score_toxicity.py \# medición de la toxicidad de las oraciones  
||-- train_eval.py \# archivo principal de ajuste fino y evaluación  
||-- utils.py  
||-- utils_eval.py  
||-- utils_format_inp.py    

|-- scripts/  \# archivos .sh    

## Conjunto de Datos


|Nombre|Categoría|Fuente|  
|----|----|------| 
| BBQ| Sesgo| https://github.com/nyu-mll/BBQ |
|HarmfulQA| Contenido Nocivo| https://huggingface.co/datasets/declare-lab/HarmfulQA |
|The Pile| Toxicidad| https://huggingface.co/datasets/tomekkorbak/pile-detoxify |
|SQuAD| Preguntas y Respuestas (QA)| https://rajpurkar.github.io/SQuAD-explorer/ |
|Alpaca| Ajuste por Instrucciones| https://huggingface.co/datasets/tatsu-lab/alpaca| 

Descarga estos datos y consulta los scripts de procesamiento correspondientes para generar datos descendentes ruidosos que contengan ejemplos seguros, no seguros y de tareas neutras. 
Estos datos descendentes ruidosos se utilizan luego para realizar ajuste fino en LLMs.  

## Evaluación  
1. *toxicidad*   
   Utilizamos [Detoxify](https://pypi.org/project/detoxify/) para medir la toxicidad.
2. *sesgo*  
   Seguimos las métricas iniciales en [BBQ](https://arxiv.org/pdf/2110.08193) para medir el sesgo.

   
## Implementaciones
### Ajuste fino 
En ``main_run.sh'', 
especifica los datos de entrenamiento en ```--train_path```.

Especifica ```--logs_dir``` para almacenar los archivos de registro y ```--save_state_path``` para guardar los estados del modelo entrenado.

Especifica ```--checkpoint_dir``` para almacenar los puntos de control durante el entrenamiento.
 Establece ```--train``` y ```--save_state``` en 1.  

 ### Evaluación del olvido 
 En ``main_run.sh'', establece ```--eval``` en 1. 
 
 Especifica ```--eval_path``` como la ruta a las completaciones pasadas del modelo y establece ```--eval_em_rouge``` en 1. 
 
 Establece ```--load_state``` en 1 y especifica ```--load_state_path``` para seleccionar el estado del modelo después del ajuste fino.  
  
### Entrenamiento intercalado  
En ``exp_interleave.sh'', especifica el estado inicial del modelo a cargar y el nombre del estado del modelo a guardar después de cada ronda de entrenamiento.  También especifica los datos de entrenamiento y el nombre del archivo de salida correspondiente para cada ronda.  
Asegúrate de que el estado inicial del modelo de una ronda sea el mismo que el nombre del estado del modelo a guardar de la ronda anterior. 

## Citación  

```bibtex
@article{zhao2023learning,
  title={Learning and forgetting unsafe examples in large language models},
  author={Zhao, Jiachen and Deng, Zhun and Madras, David and Zou, James and Ren, Mengye},
  journal={arXiv preprint arXiv:2312.12736},
  year={2023}
}
```
