
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("oof_predictions.csv")

# Largest errors in dollar terms
top5 = results.nlargest(5, "abs_error").copy()

display(top5[
    ["property_id", "address", "suburb", "bedrooms", "bathrooms",
     "parking", "land_size_sqm", "sale_price", "predicted_price",
     "error", "abs_error", "pct_error"]
].style.format({
    "sale_price": "${:,.0f}",
    "predicted_price": "${:,.0f}",
    "error": "${:,.0f}",
    "abs_error": "${:,.0f}",
    "pct_error": "{:.1f}%"
}))

# Visualise the five largest errors
plot_df = top5.sort_values("abs_error")
plt.figure(figsize=(10, 5))
plt.barh(plot_df["property_id"], plot_df["abs_error"])
plt.xlabel("Absolute prediction error ($)")
plt.ylabel("Property ID")
plt.title("Five largest out-of-fold prediction errors")
plt.tight_layout()
plt.show()

# Check whether the largest errors are unusual within their suburb
for _, row in top5.iterrows():
    peers = results[
        (results["suburb"] == row["suburb"]) &
        (results["property_id"] != row["property_id"]) &
        (results["bedrooms"] == row["bedrooms"]) &
        (results["bathrooms"] == row["bathrooms"])
    ].copy()

    if pd.notna(row["land_size_sqm"]):
        peers = peers[
            peers["land_size_sqm"].between(
                row["land_size_sqm"] - 100,
                row["land_size_sqm"] + 100
            )
        ]

    print("\n", row["property_id"], row["address"])
    print("Actual:", f"${row['sale_price']:,.0f}")
    print("Predicted:", f"${row['predicted_price']:,.0f}")
    print("Absolute error:", f"${row['abs_error']:,.0f}")
    print("Percentage error:", f"{row['pct_error']:.1f}%")
    if len(peers):
        print("Comparable-property median:",
              f"${peers['sale_price'].median():,.0f}",
              "| n =", len(peers))
    else:
        print("No close comparable properties found in the sample.")
