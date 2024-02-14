class_names = label_encoder.classes_
print(class_names)
model = grid_search.best_estimator_
#ConfusionMatrixDisplay.from_estimator(model,X,y,display_labels=class_names)

y0_test = df_test['cluster']
X_test = df_test.drop(columns = {'cluster','well'})
y_test = label_encoder.transform(y0_test)
y_pred = grid_search.predict(X_test)
print(f"Classification report for {clf_name}:")
print(classification_report(y_test, y_pred))
print("\n")

ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,display_labels=class_names)