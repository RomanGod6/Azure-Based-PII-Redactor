package main

import (
	"database/sql"
	"fmt"
	"log"
)

func main() {
	db, err := sql.Open("sqlite3", "./redactify.db")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT entity_text, entity_type, user_decision FROM training_feedback")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	fmt.Println("Training feedback records:")
	for rows.Next() {
		var entityText, entityType, userDecision string
		err := rows.Scan(&entityText, &entityType, &userDecision)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Text: %q, Type: %s, Decision: %s\n", entityText, entityType, userDecision)
	}
}
