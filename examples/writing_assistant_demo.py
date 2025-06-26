#!/usr/bin/env python3
"""
Writing Assistant Demo using AURA's Enhanced Memory System.

This script demonstrates how AURA's memory system can be used to assist
with content creation by maintaining context, retrieving relevant information,
and suggesting improvements.
"""
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType, 
    MemorySource, NamedEntity, Relation, RelationType,
    Query, ScoredMemoryNode
)


console = Console()


class WritingAssistantDemo:
    """Demo of a writing assistant using AURA's memory system."""
    
    def __init__(self):
        """Initialize the writing assistant demo."""
        self.memory_nodes = {}
        self.current_document = []
        self.document_title = "Untitled Document"
        self.document_context = {}
    
    async def create_initial_knowledge(self):
        """Create initial knowledge in the memory system."""
        console.print("\n[bold cyan]Creating initial knowledge base...[/bold cyan]")
        
        # Create source
        source = MemorySource(
            type="system",
            user_id="demo_user"
        )
        
        # Create writing tips node
        writing_tips_node = MemoryNode(
            id="mem_writing_tips_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.KNOWLEDGE_FACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="# Effective Writing Tips\n\n"
                         "1. **Start with an outline** to organize your thoughts\n"
                         "2. **Use active voice** instead of passive voice\n"
                         "3. **Be concise** and avoid unnecessary words\n"
                         "4. **Use specific examples** to illustrate your points\n"
                         "5. **Vary sentence length** to maintain reader interest\n",
                    metadata={"language": "en", "format": "markdown"}
                )
            ],
            summary="Effective writing tips",
            keywords=["writing", "tips", "style", "guide"],
            entities=[
                NamedEntity(type="concept", value="writing", confidence=0.95),
                NamedEntity(type="concept", value="style guide", confidence=0.9)
            ],
            relations=[],
            importance=0.8,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        self.memory_nodes[writing_tips_node.id] = writing_tips_node
        console.print(f"Created writing tips node: [green]{writing_tips_node.id}[/green]")
        
        # Create AI history node
        ai_history_node = MemoryNode(
            id="mem_ai_history_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.KNOWLEDGE_FACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="# Brief History of AI\n\n"
                         "Artificial Intelligence (AI) has evolved significantly since its inception in the 1950s:\n\n"
                         "- **1950s**: Alan Turing proposes the Turing Test\n"
                         "- **1956**: The term 'Artificial Intelligence' is coined at Dartmouth Conference\n"
                         "- **1960s-70s**: Early AI systems like ELIZA and expert systems emerge\n"
                         "- **1980s**: Machine learning begins to gain traction\n"
                         "- **1990s-2000s**: Statistical methods and neural networks advance\n"
                         "- **2010s**: Deep learning revolution with ImageNet competition\n"
                         "- **2020s**: Large language models like GPT and multimodal AI systems\n",
                    metadata={"language": "en", "format": "markdown"}
                )
            ],
            summary="Brief history of artificial intelligence",
            keywords=["AI", "history", "timeline", "development"],
            entities=[
                NamedEntity(type="concept", value="artificial intelligence", confidence=0.95),
                NamedEntity(type="person", value="Alan Turing", confidence=0.9),
                NamedEntity(type="event", value="Dartmouth Conference", confidence=0.85)
            ],
            relations=[],
            importance=0.75,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        self.memory_nodes[ai_history_node.id] = ai_history_node
        console.print(f"Created AI history node: [green]{ai_history_node.id}[/green]")
        
        # Create AI applications node
        ai_applications_node = MemoryNode(
            id="mem_ai_applications_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.KNOWLEDGE_FACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="# Modern AI Applications\n\n"
                         "AI is transforming numerous industries today:\n\n"
                         "- **Healthcare**: Medical diagnosis, drug discovery, personalized treatment\n"
                         "- **Finance**: Fraud detection, algorithmic trading, risk assessment\n"
                         "- **Transportation**: Self-driving vehicles, traffic optimization\n"
                         "- **Retail**: Recommendation systems, inventory management\n"
                         "- **Manufacturing**: Predictive maintenance, quality control\n"
                         "- **Entertainment**: Content creation, personalized recommendations\n"
                         "- **Education**: Adaptive learning, automated grading\n",
                    metadata={"language": "en", "format": "markdown"}
                )
            ],
            summary="Modern applications of artificial intelligence",
            keywords=["AI", "applications", "industries", "technology"],
            entities=[
                NamedEntity(type="concept", value="artificial intelligence", confidence=0.95),
                NamedEntity(type="industry", value="healthcare", confidence=0.9),
                NamedEntity(type="industry", value="finance", confidence=0.9),
                NamedEntity(type="industry", value="transportation", confidence=0.9)
            ],
            relations=[],
            importance=0.8,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        self.memory_nodes[ai_applications_node.id] = ai_applications_node
        console.print(f"Created AI applications node: [green]{ai_applications_node.id}[/green]")
        
        # Create AI ethics node
        ai_ethics_node = MemoryNode(
            id="mem_ai_ethics_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.KNOWLEDGE_FACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="# Ethical Considerations in AI\n\n"
                         "As AI becomes more powerful, several ethical concerns arise:\n\n"
                         "- **Bias and Fairness**: AI systems can perpetuate or amplify existing biases\n"
                         "- **Privacy**: AI often requires vast amounts of data, raising privacy concerns\n"
                         "- **Transparency**: Many AI systems are 'black boxes' with unclear decision processes\n"
                         "- **Accountability**: Determining responsibility when AI systems cause harm\n"
                         "- **Job Displacement**: Automation may eliminate certain types of jobs\n"
                         "- **Security**: AI systems can be vulnerable to adversarial attacks\n"
                         "- **Autonomy**: Questions about human control over increasingly autonomous systems\n",
                    metadata={"language": "en", "format": "markdown"}
                )
            ],
            summary="Ethical considerations in artificial intelligence",
            keywords=["AI", "ethics", "bias", "privacy", "transparency"],
            entities=[
                NamedEntity(type="concept", value="artificial intelligence", confidence=0.95),
                NamedEntity(type="concept", value="ethics", confidence=0.95),
                NamedEntity(type="concept", value="bias", confidence=0.9),
                NamedEntity(type="concept", value="privacy", confidence=0.9)
            ],
            relations=[],
            importance=0.85,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        self.memory_nodes[ai_ethics_node.id] = ai_ethics_node
        console.print(f"Created AI ethics node: [green]{ai_ethics_node.id}[/green]")
        
        # Add relationships between nodes
        ai_history_node.relations.append(Relation(
            type=RelationType.SIMILAR_TO,  # Using SIMILAR_TO instead of RELATED_TO
            target_id=ai_applications_node.id,
            strength=0.7
        ))
        
        ai_history_node.relations.append(Relation(
            type=RelationType.REFERENCES,  # Using REFERENCES instead of RELATED_TO
            target_id=ai_ethics_node.id,
            strength=0.6
        ))
        
        ai_applications_node.relations.append(Relation(
            type=RelationType.PART_OF,  # Using PART_OF instead of RELATED_TO
            target_id=ai_ethics_node.id,
            strength=0.8
        ))
        
        console.print("\n[bold green]Knowledge base created successfully![/bold green]")
    
    async def search_memory(self, query: str) -> List[ScoredMemoryNode]:
        """Search memory for relevant information.
        
        Args:
            query: Search query
            
        Returns:
            List of scored memory nodes
        """
        console.print(f"\n[bold]Searching memory for:[/bold] {query}")
        
        # Simple keyword-based search for demo purposes
        results = []
        for node_id, node in self.memory_nodes.items():
            # Calculate simple relevance score based on keyword matching
            score = 0.0
            query_terms = query.lower().split()
            
            # Check keywords
            for term in query_terms:
                if any(term in keyword.lower() for keyword in node.keywords):
                    score += 0.2
            
            # Check summary
            if any(term in node.summary.lower() for term in query_terms):
                score += 0.3
            
            # Check content
            text_content = node.get_text_content().lower()
            for term in query_terms:
                if term in text_content:
                    score += 0.5
            
            # Normalize score
            score = min(score, 1.0)
            
            # Add to results if relevant
            if score > 0.1:
                results.append(ScoredMemoryNode(
                    node=node,
                    score=score
                ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update access stats for retrieved nodes
        for result in results:
            result.node.update_access_stats()
        
        return results
    
    async def get_writing_suggestions(self, current_text: str) -> Dict[str, Any]:
        """Get writing suggestions based on current text.
        
        Args:
            current_text: Current document text
            
        Returns:
            Dictionary of suggestions
        """
        # Extract key terms from current text
        terms = set(current_text.lower().split())
        important_terms = [term for term in terms if len(term) > 4]
        
        # Determine document topic
        if "ai" in terms or "artificial" in terms or "intelligence" in terms:
            topic = "artificial intelligence"
        elif "writing" in terms or "document" in terms:
            topic = "writing"
        else:
            topic = "general"
        
        # Search for relevant information
        query = f"{topic} {' '.join(important_terms[:5])}"
        results = await self.search_memory(query)
        
        # Generate suggestions
        suggestions = {
            "topic": topic,
            "key_terms": important_terms[:5],
            "style_suggestions": [],
            "content_suggestions": [],
            "related_information": []
        }
        
        # Add style suggestions
        if len(current_text.split()) > 10:
            # Check for passive voice (simplified)
            if " is " in current_text and " by " in current_text:
                suggestions["style_suggestions"].append(
                    "Consider using active voice instead of passive voice"
                )
            
            # Check for sentence variety
            sentences = current_text.split(". ")
            if len(sentences) > 2:
                sentence_lengths = [len(s.split()) for s in sentences]
                if max(sentence_lengths) - min(sentence_lengths) < 3:
                    suggestions["style_suggestions"].append(
                        "Try varying your sentence length for better rhythm"
                    )
        
        # Add content suggestions from search results
        for result in results[:2]:
            content = result.node.get_text_content()
            suggestions["content_suggestions"].append({
                "source": result.node.summary,
                "relevance": result.score,
                "content": content[:200] + "..." if len(content) > 200 else content
            })
        
        # Add related information
        for result in results:
            for relation in result.node.relations:
                if relation.target_id in self.memory_nodes:
                    related_node = self.memory_nodes[relation.target_id]
                    suggestions["related_information"].append({
                        "title": related_node.summary,
                        "relation_type": relation.type.value,
                        "strength": relation.strength
                    })
        
        return suggestions
    
    async def run_demo(self):
        """Run the writing assistant demo."""
        console.print(Panel(
            "[bold]AURA Writing Assistant Demo[/bold]\n"
            "This demo shows how AURA's memory system can assist with content creation",
            border_style="cyan"
        ))
        
        # Create initial knowledge
        await self.create_initial_knowledge()
        
        # Set document title
        self.document_title = Prompt.ask(
            "\n[bold]Enter document title[/bold]",
            default="The Future of Artificial Intelligence"
        )
        
        console.print(f"\n[bold]Creating document:[/bold] {self.document_title}")
        
        # Main writing loop
        while True:
            # Display current document
            if self.current_document:
                console.print("\n[bold]Current Document:[/bold]")
                document_text = "\n".join(self.current_document)
                console.print(Markdown(f"# {self.document_title}\n\n{document_text}"))
            else:
                console.print("\n[bold]Document is empty. Start writing![/bold]")
            
            # Get user input
            user_input = Prompt.ask(
                "\n[bold]Enter text to add (or 'q' to quit, 's' for suggestions)[/bold]"
            )
            
            if user_input.lower() == 'q':
                break
            
            if user_input.lower() == 's':
                # Get suggestions based on current document
                document_text = "\n".join(self.current_document)
                suggestions = await self.get_writing_suggestions(document_text)
                
                # Display suggestions
                console.print("\n[bold cyan]Writing Suggestions:[/bold cyan]")
                
                # Style suggestions
                if suggestions["style_suggestions"]:
                    console.print("\n[bold]Style Suggestions:[/bold]")
                    for suggestion in suggestions["style_suggestions"]:
                        console.print(f"• {suggestion}")
                
                # Content suggestions
                if suggestions["content_suggestions"]:
                    console.print("\n[bold]Relevant Content:[/bold]")
                    for content in suggestions["content_suggestions"]:
                        panel = Panel(
                            Markdown(content["content"]),
                            title=f"[bold]{content['source']}[/bold] (Relevance: {content['relevance']:.2f})",
                            border_style="green"
                        )
                        console.print(panel)
                
                # Related information
                if suggestions["related_information"]:
                    console.print("\n[bold]Related Topics:[/bold]")
                    table = Table()
                    table.add_column("Topic", style="cyan")
                    table.add_column("Relation", style="green")
                    table.add_column("Strength", style="yellow")
                    
                    for info in suggestions["related_information"]:
                        table.add_row(
                            info["title"],
                            info["relation_type"],
                            f"{info['strength']:.2f}"
                        )
                    
                    console.print(table)
            else:
                # Add text to document
                self.current_document.append(user_input)
                
                # Create memory node for this paragraph
                paragraph_node = MemoryNode(
                    id=f"mem_paragraph_{len(self.current_document)}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(
                        type="user_input",
                        user_id="demo_user"
                    ),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=user_input,
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Paragraph {len(self.current_document)} of document",
                    keywords=user_input.lower().split()[:5],
                    entities=[],
                    relations=[],
                    importance=0.7,
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                
                self.memory_nodes[paragraph_node.id] = paragraph_node
        
        # Save final document
        if self.current_document:
            console.print("\n[bold green]Final Document:[/bold green]")
            document_text = "\n".join(self.current_document)
            console.print(Markdown(f"# {self.document_title}\n\n{document_text}"))
            
            # Save to file
            output_dir = Path("/tmp/aura_writing_demo")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{self.document_title.replace(' ', '_')}.md"
            
            with open(output_file, "w") as f:
                f.write(f"# {self.document_title}\n\n{document_text}")
            
            console.print(f"\n[bold]Document saved to:[/bold] {output_file}")
        
        console.print("\n[bold green]Demo completed successfully![/bold green]")


async def main():
    """Main function."""
    demo = WritingAssistantDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())