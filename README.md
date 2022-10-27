# Разработка системы машинного обучения
Лабораторная работа №3 для АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ

Отчет по лабораторной работе #3 выполнил(а):
- Голубятникова Ксения Александровна
- РИ210939

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;


## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets - Unity
- Создание проекта на Unity
- Добавление в проект ML Agent

![image](https://user-images.githubusercontent.com/114469025/197522371-8ac1cdad-723c-43da-ad16-0ac0a6c53a42.png)
![image](https://user-images.githubusercontent.com/114469025/197621955-9402bcf2-27e7-4a86-a14e-df9713028f40.png)


- Написание команд в Anaconda Prompt для создания и активации нового ML-агента и добавления библиотек
![image](https://user-images.githubusercontent.com/114469025/197525153-4e1d8b86-20b9-406a-b4e7-61d27080d0ba.png)

![image](https://user-images.githubusercontent.com/114469025/197525206-41d07b8f-48d0-4a7d-ab9d-b37a29f16450.png)

![image](https://user-images.githubusercontent.com/114469025/197614732-6e3dcc08-290b-414c-933b-5c0ecd427d3d.png)
![image](https://user-images.githubusercontent.com/114469025/197622320-e7ec8532-8bd6-4a9c-a954-e44c23669e1b.png)



- Создание на сцене плоскости, сферы и кубика

![image](https://user-images.githubusercontent.com/114469025/197623363-a7313551-0157-4be0-811c-3ee03383905b.png)
![image](https://user-images.githubusercontent.com/114469025/197624428-ea0d3c11-056c-4fff-8fdd-87c158ef988e.png)

- Создание С#-скрипта и подключение нено к сцене
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
![image](https://user-images.githubusercontent.com/114469025/197625540-59ba7b9f-72fd-426e-a257-5843a772177e.png)

- Добавление файла конфигурации нейронной сети
```py
behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```py
![image](https://user-images.githubusercontent.com/114469025/197630325-35614f16-e9ea-47e7-88c1-7120e9204095.png)

- Запуск работы ml-агента

![image](https://user-images.githubusercontent.com/114469025/198267575-a68a2440-3c55-492e-8910-7e0d0022f893.png)
![image](https://user-images.githubusercontent.com/114469025/198267604-fcabcd42-3a7a-4054-b4f0-a340d201eedf.png)
![image](https://user-images.githubusercontent.com/114469025/198267695-c12fc631-7a79-4987-ac5d-b87f30af6fbd.png)

![image](https://user-images.githubusercontent.com/114469025/198270029-1be2fd3d-bbda-45fe-9687-d36879c3c051.png)
![image](https://user-images.githubusercontent.com/114469025/198270062-1682cf33-e881-467d-9153-e4676d6f72bd.png)


![image](https://user-images.githubusercontent.com/114469025/198272275-81ab5208-a940-4f29-8a4a-fb6d3243c575.png)


![image](https://user-images.githubusercontent.com/114469025/198273455-a7932726-ff39-4945-b1fe-1c2c5c31315f.png)
![image](https://user-images.githubusercontent.com/114469025/198273514-6148f3a4-c30f-476d-a18e-d1676c294ddf.png)


- Проверка работы модели




## Задание 2
### Описание строк файла конфигурации нейронной сети
- Decision Requester - это компонент запроса решений, который автоматически запрашивает решения для экземпляра через регулярные промежутки времени. Предоставляет удобный и гибкий запуска процесса принятия решения агентом.
- Behavior Parametrs - у каждого агента должно быть определенное поведение. Behavior Parametrs определяет, как Агент принимает решения. Например в RollerAgent используется пробел размером 8. Это означает, что вектор признаков, содержащий наблюдения Агента, содержит восемь элементов: компоненты x и z вращения куба агента и компоненты x, y и z относительного положения и скорости шара.


- trainer_type: ppo Тип используемого тренажера: ppo, sac или poca.
- hyperparameters:
      batch_size: 10 Количество демонстрационных опытов, использованных для одной итерации обновления градиентного спуска. 
      buffer_size: 100 Количество опытов на каждой итерации градиентного спуска. Это всегда должно быть в несколько раз меньше, чем buffer_size. 
      learning_rate: 3.0e-4 Скорость обучения, используемая для обновления дискриминатора. 
      beta: 5.0e-4 Сила регуляризации энтропии, которая делает политику "более случайной". Это гарантирует, что агенты должным образом исследуют пространство действий во время обучения. Увеличение этого параметра обеспечит выполнение большего количества случайных действий. Это должно быть скорректировано таким образом, чтобы энтропия медленно уменьшалась вместе с увеличением вознаграждения. Если энтропия падает слишком быстро, увеличьте бета-версию. Если энтропия падает слишком медленно, уменьшите бета.
      epsilon: 0.2 Влияет на то, насколько быстро политика может развиваться во время обучения. Соответствует допустимому порогу расхождения между старой и новой политиками при обновлении с градиентным спуском. Установка этого малого значения приведет к более стабильным обновлениям, но также замедлит процесс обучения.
      lambd: 0.99 Параметр регуляризации (лямбда), используемый при расчете обобщенной оценки преимущества. Это можно рассматривать как то, насколько агент полагается на свою текущую оценку стоимости при расчете обновленной оценки стоимости.
      learning_rate_schedule: linear Определяет, как скорость обучения меняется с течением времени. 
    network_settings:
      normalize: false Применяется к входным данным векторного наблюдения. Эта нормализация основана на текущем среднем значении и дисперсии векторного наблюдения. 
      hidden_units: 128  Количество единиц в скрытых слоях нейронной сети. Соответствует количеству единиц в каждом полностью связном слое нейронной сети. 
      num_layers: 2 Количество скрытых слоев в нейронной сети. Соответствует тому, сколько скрытых слоев присутствует после ввода наблюдения или после кодирования CNN визуального наблюдения. 
      
    reward_signals:
      extrinsic:
        gamma: 0.99 Коэффициент дисконтирования для будущих вознаграждений, поступающих от окружающей среды. Это можно рассматривать как то, насколько далеко в будущем агент должен заботиться о возможных вознаграждениях. 
        strength: 1.0 Фактор, на который можно умножить вознаграждение, получаемое от окружающей среды. 
    max_steps: 500000 Общее количество шагов (т.е. собранных наблюдений и предпринятых действий), которые должны быть выполнены в среде (или во всех средах, если используется несколько параллельно) до завершения процесса обучения.
    
    time_horizon: 64 количество шагов опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта. Когда этот предел достигается до окончания эпизода, оценка стоимости используется для прогнозирования общего ожидаемого вознаграждения от текущего состояния агента.
    
- summary_freq: 10000 Количество опыта, который необходимо собрать перед созданием и отображением статистики обучения. 

Игровой баланс - это 
