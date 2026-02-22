# Lab 12: Turniej Modeli — ewaluacja i finał

## Czy wiesz, że...

W 2014 roku grupa badaczy odkryła, że większość publikacji o predykcji defektów używa accuracy jako głównej metryki — mimo że datasety są niezbalansowane. Po przeliczeniu wyników na AUC-ROC okazało się, że połowa "przełomowych" modeli była nie lepsza niż rzut monetą. Od tamtego czasu każda szanująca się konferencja wymaga raportowania wielu metryk.

## Kontekst

Na lab11 wytrenowaliście trzy modele i porównaliście je na zbiorze testowym. Ale jedno porównanie na jednym podziale danych to za mało — wyniki mogą być przypadkowe. Dziś robimy to porządnie: cross-validation, krzywe ROC, i raport końcowy podsumowujący całą podróż od metryk do predykcji.

To ostatnie laboratorium w kursie. Czas zamknąć koło: od ręcznego `git log` na labie 1, przez budowanie datasetu, aż po trenowanie i ewaluację modeli ML.

## Cel laboratorium

Po tym laboratorium będziesz potrafić:
- obliczać i interpretować metryki ewaluacji: precision, recall, F1, AUC-ROC,
- przeprowadzać k-fold cross-validation w scikit-learn,
- rysować i interpretować krzywe ROC,
- porównywać modele i wybierać najlepszy z uzasadnieniem.

## Wymagania wstępne

- Python 3.9+ z bibliotekami: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
- Jupyter Notebook
- Plik `dataset.csv` z lab10 (lub przykładowy dataset z lab11)
- Wytrenowane modele z lab11 (lub wytrenujecie je ponownie w notebooku)

## Zadania

### Zadanie 1: Metryki ewaluacji (30 min)

Na lab11 obliczyliście accuracy i confusion matrix. Dziś idziemy głębiej.

**Krok 1:** Otwórzcie szablon `model_tournament.ipynb` z tego repozytorium.

**Krok 2:** Wczytajcie dataset i wytrenujcie 3 modele (jak na lab11) — w notebooku jest gotowy kod.

**Krok 3:** Dla każdego modelu obliczcie dodatkowe metryki:

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

# Precision, Recall, F1 (dla klasy buggy)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# AUC-ROC (wymaga prawdopodobieństw, nie predykcji)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
```

**Krok 4:** Narysujcie krzywe ROC dla wszystkich 3 modeli na jednym wykresie:

```python
fig, ax = plt.subplots(figsize=(8, 6))

for name, model, X_eval in [
    ("Regresja logistyczna", lr, X_test_scaled),
    ("Drzewo decyzyjne", dt, X_test),
    ("Random Forest", rf, X_test),
]:
    y_proba = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Krzywe ROC")
ax.legend()
plt.tight_layout()
plt.show()
```

**Krok 5:** Odpowiedzcie:
1. Który model ma najwyższe AUC-ROC? Czy to ten sam, który miał najwyższe accuracy?
2. Co oznacza AUC = 0.5? A AUC = 1.0?
3. Dlaczego AUC-ROC jest lepszą metryką niż accuracy dla niezbalansowanych datasetów?

### Zadanie 2: Cross-validation i porównanie (60 min)

Jeden podział train/test to za mało. Cross-validation daje bardziej wiarygodne wyniki.

**Krok 1:** Przeprowadźcie 10-fold stratified cross-validation:

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Dla każdego modelu:
# cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
```

**Krok 2:** Zbierzcie wyniki w tabelę porównawczą:

| Metryka | Regresja log. | Drzewo dec. | Random Forest |
|---------|--------------|-------------|---------------|
| Accuracy | mean +/- std | mean +/- std | mean +/- std |
| Precision | mean +/- std | mean +/- std | mean +/- std |
| Recall | mean +/- std | mean +/- std | mean +/- std |
| F1 | mean +/- std | mean +/- std | mean +/- std |
| AUC-ROC | mean +/- std | mean +/- std | mean +/- std |

**Krok 3:** Wizualizacja porównania — boxploty wyników cross-validation:

```python
# Boxploty F1 score z cross-validation
fig, ax = plt.subplots(figsize=(10, 5))
results_f1 = [cv_lr["test_f1"], cv_dt["test_f1"], cv_rf["test_f1"]]
ax.boxplot(results_f1, labels=["Logistic Regression", "Decision Tree", "Random Forest"])
ax.set_ylabel("F1 Score")
ax.set_title("Porównanie modeli (10-fold CV)")
plt.tight_layout()
plt.show()
```

**Krok 4:** Odpowiedzcie:
1. Który model ma najwyższy średni F1? A najwyższy AUC-ROC?
2. Który model ma najniższe odchylenie standardowe (jest najbardziej stabilny)?
3. Czy ranking modeli zmienił się w porównaniu z wynikami z lab11 (pojedynczy split)?
4. Czy różnice między modelami są duże czy małe? Czy to ma praktyczne znaczenie?

### Zadanie 3: Raport końcowy (45 min)

Czas na podsumowanie całej podróży od lab01 do lab12.

**Napiszcie raport (w notebooku jako komórki Markdown lub w osobnym `report.md`):**

1. **Projekt:** Jaki projekt OSS analizowaliście? Dlaczego go wybraliście?

2. **Dataset:** Jak budowaliście dataset?
   - Ile plików, ile cech?
   - Jak etykietowaliście (heurystyka)?
   - Czy dataset był zbalansowany?

3. **Feature importance:** Które metryki kodu najlepiej predykują defekty?
   - Top 3 cechy wg Random Forest
   - Czy to zgadza się z intuicją? Z literaturą?

4. **Najlepszy model:** Który model wybralibyście do produkcji?
   - Dlaczego ten, a nie inny?
   - Jaką metrykę optymalizowalibyście (accuracy, recall, F1, AUC)?
   - Jakie są jego ograniczenia?

5. **Ograniczenia:**
   - Heurystyka etykietowania — jakie są jej wady?
   - Mały dataset — ile próbek to "za mało" dla ML?
   - Generalizowalność — czy model wytrenowany na requests zadziała na flask?
   - Co zrobilibyście inaczej, gdybyście mieli więcej czasu?

6. **Wnioski z kursu:**
   - Co zapamiętaliście z 12 labów?
   - Które narzędzie/metryka okazały się najbardziej użyteczne?
   - Jak mierzenie kodu wpłynęło na wasze postrzeganie jakości oprogramowania?

## Co oddajecie

W swoim branchu `lab12_nazwisko1_nazwisko2`:

1. **`model_tournament.ipynb`** — wypełniony notebook z ewaluacją i cross-validation
2. **Raport końcowy** — w notebooku lub w osobnym `report.md`

## Kryteria oceny

- Obliczone metryki ewaluacji (precision, recall, F1, AUC-ROC) dla 3 modeli
- Krzywe ROC na jednym wykresie
- 10-fold cross-validation z tabelą porównawczą (mean +/- std)
- Boxploty wyników cross-validation
- Raport końcowy z odpowiedziami na wszystkie 6 punktów
- Sensowne wnioski i uzasadnienie wyboru modelu

## FAQ

**P: AUC-ROC wychodzi mi 0.5 — czy model jest zepsuty?**
O: AUC = 0.5 oznacza, że model nie jest lepszy niż losowe zgadywanie. Możliwe przyczyny: za mało danych, cechy nie rozróżniają klas, lub model nie trenuje się poprawnie. Sprawdź czy dane są poprawnie przygotowane.

**P: Cross-validation trwa bardzo długo.**
O: Zmniejsz liczbę foldów (z 10 na 5) lub liczbę drzew w Random Forest (z 100 na 50). Dla małych datasetów CV powinno trwać sekundy.

**P: Wyniki cross-validation mają duże odchylenie standardowe.**
O: To normalne przy małych datasetach. Każdy fold ma mało próbek, więc wyniki się wahają. Zanotujcie to jako obserwację.

**P: Nie wiem, który model wybrać do produkcji.**
O: Nie ma jednej "prawdziwej" odpowiedzi. Zależy od tego, co jest ważniejsze: precision (mniej fałszywych alarmów) czy recall (mniej przegapionych bugów). Uzasadnijcie swój wybór.

**P: Czy muszę pisać raport końcowy od zera?**
O: Nie — możecie go napisać jako komórki Markdown w notebooku. Nie musi być długi — 1-2 strony tekstu z konkretnymi odpowiedziami na 6 punktów.

**P: Co jeśli moje wyniki są "złe" (niskie F1, AUC bliskie 0.5)?**
O: To nie jest problem — to obserwacja naukowa! Zanotujcie to i wyjaśnijcie dlaczego (mały dataset, słabe cechy, niezbalansowane klasy). Uczciwa analiza słabych wyników jest ważniejsza niż ładne liczby.

## Przydatne linki

- [ROC Curve (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Cross-validation (scikit-learn)](https://scikit-learn.org/stable/modules/cross_validation.html)
- [AUC-ROC explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Precision-Recall tradeoff](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Stratified K-Fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

---
*"Hunger Games, ale dla algorytmów. Niech zwycięży najlepszy model! ...a potem i tak użyjemy Random Forest."* — anonimowy data scientist
