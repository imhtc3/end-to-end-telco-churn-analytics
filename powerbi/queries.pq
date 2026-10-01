// ---------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------
DataFolder = "C:\Projects\telco-churn-analytics\data\powerbi\" meta [IsParameterQuery = true, Type = "Text", IsParameterQueryRequired = true]


// ---------------------------------------------------------------------
// fnLoadCsv - shared loader, promotes headers and trims text columns
// ---------------------------------------------------------------------
fnLoadCsv = (fileName as text) as table =>
let
    Source   = Csv.Document(File.Contents(DataFolder & fileName), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    // everything arrives as text at this point, typing happens per table
    Trimmed  = Table.TransformColumns(Promoted, List.Transform(Table.ColumnNames(Promoted), each {_, Text.Trim, type text}))
in
    Trimmed


// ---------------------------------------------------------------------
// fact_subscription
// ---------------------------------------------------------------------
fact_subscription =
let
    Source = fnLoadCsv("fact_subscription.csv"),
    Typed  = Table.TransformColumnTypes(Source, {
        {"customer_id", type text}, {"contract_key", Int64.Type}, {"payment_key", Int64.Type},
        {"internet_key", Int64.Type}, {"tenure", Int64.Type}, {"tenure_band", type text},
        {"phone_service", Int64.Type}, {"multiple_lines", type text}, {"paperless_billing", Int64.Type},
        {"n_addons", Int64.Type}, {"n_services", Int64.Type}, {"monthly_charges", Currency.Type},
        {"total_charges", Currency.Type}, {"charge_drift", type number}, {"churn", Int64.Type}
    }, "en-US"),
    Status = Table.AddColumn(Typed, "status", each if [churn] = 1 then "Churned" else "Active", type text)
in
    Status


// ---------------------------------------------------------------------
// dim_customer
// ---------------------------------------------------------------------
dim_customer =
let
    Source = fnLoadCsv("dim_customer.csv"),
    Typed  = Table.TransformColumnTypes(Source, {
        {"customer_id", type text}, {"gender", type text}, {"senior_citizen", Int64.Type},
        {"partner", Int64.Type}, {"dependents", Int64.Type}, {"family", Int64.Type}
    }),
    Senior = Table.AddColumn(Typed, "age_group", each if [senior_citizen] = 1 then "Senior" else "Under 65", type text),
    Household = Table.AddColumn(Senior, "household", each
        if [partner] = 1 and [dependents] = 1 then "Partner + dependents"
        else if [partner] = 1 then "Partner"
        else if [dependents] = 1 then "Dependents"
        else "Single", type text)
in
    Household


// ---------------------------------------------------------------------
// small dimensions
// ---------------------------------------------------------------------
dim_contract =
let
    Source = fnLoadCsv("dim_contract.csv"),
    Typed  = Table.TransformColumnTypes(Source, {{"contract_key", Int64.Type}, {"contract_type", type text}, {"contract_months", Int64.Type}})
in
    Typed

dim_payment =
let
    Source = fnLoadCsv("dim_payment.csv"),
    Typed  = Table.TransformColumnTypes(Source, {{"payment_key", Int64.Type}, {"payment_method", type text}, {"auto_pay", Int64.Type}}),
    Short  = Table.AddColumn(Typed, "payment_short", each Text.BeforeDelimiter([payment_method], " ("), type text)
in
    Short

dim_internet =
let
    Source = fnLoadCsv("dim_internet.csv"),
    Typed  = Table.TransformColumnTypes(Source, {{"internet_key", Int64.Type}, {"internet_service", type text}})
in
    Typed

dim_tenure_band =
let
    Source = fnLoadCsv("dim_tenure_band.csv"),
    Typed  = Table.TransformColumnTypes(Source, {{"tenure_band", type text}, {"band_order", Int64.Type}})
in
    Typed

dim_segment =
let
    Source = fnLoadCsv("dim_segment.csv"),
    Typed  = Table.TransformColumnTypes(Source, {{"customer_id", type text}, {"segment_id", Int64.Type}, {"segment_name", type text}})
in
    Typed


// ---------------------------------------------------------------------
// bridge_customer_service
// ---------------------------------------------------------------------
bridge_customer_service =
let
    Source  = fnLoadCsv("bridge_customer_service.csv"),
    Typed   = Table.TransformColumnTypes(Source, {{"customer_id", type text}, {"service_name", type text}, {"status", type text}}),
    Pretty  = Table.AddColumn(Typed, "service", each Text.Proper(Text.Replace([service_name], "_", " ")), type text)
in
    Pretty


// ---------------------------------------------------------------------
// fact_churn_score (output of the Python model)
// ---------------------------------------------------------------------
fact_churn_score =
let
    Source = fnLoadCsv("fact_churn_score.csv"),
    Typed  = Table.TransformColumnTypes(Source, {
        {"customer_id", type text}, {"model_name", type text}, {"churn_probability", type number},
        {"risk_tier", type text}, {"scored_at", type datetimezone}, {"split", type text}, {"targeted", Int64.Type}
    }, "en-US"),
    TierOrder = Table.AddColumn(Typed, "tier_order", each
        List.PositionOf({"Low", "Medium", "High", "Critical"}, [risk_tier]) + 1, Int64.Type),
    Bucket = Table.AddColumn(TierOrder, "probability_bucket", each
        Number.RoundDown([churn_probability] * 10) / 10, type number)
in
    Bucket
