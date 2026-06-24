def add_ticket_group_size(df):
    df = df.copy()
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    return df


def add_fare_per_ticket_member(df):
    df = df.copy()

    if "TicketGroupSize" not in df.columns:
        df = add_ticket_group_size(df)

    df["FarePerTicketMember"] = df["Fare"] / df["TicketGroupSize"]
    return df