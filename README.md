# MIPT Lab: OAT + BLT / H-Net

Расширения поверх [OAT](https://github.com/Chaoqi-LIU/oat) (vendor в `third_party/oat`).

## Структура

| Путь | Назначение |
|------|------------|
| `third_party/oat` | Upstream OAT (клон + submodule LIBERO) |
| `src/oat_ext` | Ваш код: динамический патчинг, иерархия, merge конфигов |
| `configs/hydra_overrides` | Черновики Hydra-overrides для ablation |
| `scripts/` | Установка, обучение baseline, eval |
| `experiments/` | Логи и чекпоинты (не коммитить большие файлы) |
| `tests/` | Юнит-тесты для `oat_ext` |

## Быстрый старт

1. Установить [uv](https://docs.astral.sh/uv/) (если ещё нет): скрипт установки кладёт бинарник в `~/.local/bin` — добавьте его в `PATH` в `~/.zshrc`.

2. Подтянуть submodule и установить OAT:

   ```bash
   ./scripts/install_oat.sh
   ```

   Скрипт ставит пакет `cmake` в venv OAT и задаёт `CMAKE_POLICY_VERSION_MINIMUM=3.5` — иначе на macOS со **CMake 4.x** сборка `egl-probe` (зависимость `robomimic`) падает.

3. Данные **libero10** в формате zarr (ожидается путь `third_party/oat/data/libero/libero10_N500.zarr`, см. конфиг `task/tokenizer`):

   ```bash
   chmod +x ./scripts/download_libero10_zarr.sh
   ./scripts/download_libero10_zarr.sh
   ```

   Если Hugging Face отвечает **429**, повторите позже или скачайте архив вручную по ссылке из [README OAT](https://github.com/Chaoqi-LIU/oat) и распакуйте в `third_party/oat/data/libero/`. Альтернатива — сборка zarr из HDF5 по инструкции апстрима.

4. Обучить OAT tokenizer (baseline) **на GPU**.

   Обучение токенизатора — это тяжёлый PyTorch-процесс: **нормальный режим — CUDA-GPU** (как в README апстрима с `accelerate launch --multi_gpu`). На CPU запуск возможен, но будет на порядки медленнее и годится разве что на smoke-тест. Убедитесь, что видна видеокарта (`nvidia-smi` на Linux, на macOS обычно нет CUDA — используйте машину с NVIDIA или облако).

   Одна карта, явно выбрать устройство 0:

   ```bash
   CUDA_VISIBLE_DEVICES=0 ./scripts/train_baseline.sh
   ```

   Полный запуск:

   ```bash
   ./scripts/train_baseline.sh
   ```

   Короткий пробный прогон (после появления zarr):

   ```bash
   ./scripts/train_baseline.sh training.num_epochs=1 training.val_every=1 dataloader.batch_size=4
   ```

   Дополнительные Hydra-аргументы передаются в конец, например: `training.num_epochs=50`.

5. Тесты расширений:

   ```bash
   pip install -r requirements.txt
   pytest
   ```

## Eval политики

После обучения policy и наличия `.ckpt`:

```bash
./scripts/eval_libero.sh /path/to/oatpolicy.ckpt --num_exp 5
```

## Примечание

Апстрим использует `train_oattok` и `scripts/run_workspace.py`, а не отдельный `train_tokenizer.py` в корне — команды в скриптах соответствуют актуальному README OAT.

### English: early exit (test assignment track)

For the **early-exit gate** on OAT policy token generation (MLP + optional max-prob heuristic), see **[docs/EARLY_EXIT.md](docs/EARLY_EXIT.md)**. Offline training with reconstruction labels: **`scripts/train_early_exit_offline.py`**. Report template: **[docs/EXPERIMENTS_SECTION_TEMPLATE.md](docs/EXPERIMENTS_SECTION_TEMPLATE.md)**.
