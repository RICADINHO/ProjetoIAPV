# ProjetoIAPV

Biblioteca necessarias para funcionamento:

pip install gymnasium -> Para a criação de um novo ambiente (gym)

pip install stable-baselines3 -> Para a policy no GAIL

pip install imitation -> Para os algoritmos de imitação

Instrução para testar o train.py:
- Para o BC: python train.py --file="demo_1.pkl" --gym=”CartPole” --algorithm=”BC” --output=”policy.zip”
- Para o GAIL: python train.py --file="demo_1.pkl" --gym=”CartPole” --algorithm=”GAIL” --output=”policy.zip”