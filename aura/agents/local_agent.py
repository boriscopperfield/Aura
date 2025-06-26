"""
Local Agent implementation for local tools and services.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import shutil

from .base import Agent, AgentCapability, AgentResult
from ..utils.errors import AgentError


class LocalAgent(Agent):
    """Local tool execution agent."""
    
    def __init__(
        self,
        name: str = "local",
        tools_dir: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            capabilities=[
                AgentCapability.CODE_GENERATION,  # Local code execution
                AgentCapability.ANALYSIS,         # File analysis
                AgentCapability.IMAGE_ANALYSIS    # Local image processing
            ],
            **kwargs
        )
        
        self.tools_dir = tools_dir or Path("/usr/local/bin")
        self.temp_dir = Path(tempfile.gettempdir()) / "aura_local"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute local capability."""
        if capability == AgentCapability.CODE_GENERATION:
            return await self._execute_code(**kwargs)
        elif capability == AgentCapability.ANALYSIS:
            return await self._analyze_file(**kwargs)
        elif capability == AgentCapability.IMAGE_ANALYSIS:
            return await self._process_image(**kwargs)
        else:
            return AgentResult.error_result(
                f"Capability {capability.value} not implemented",
                0.0
            )
    
    async def _execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        **kwargs
    ) -> AgentResult:
        """Execute code locally."""
        try:
            # Create temporary file
            if language == "python":
                suffix = ".py"
                command = ["python3"]
            elif language == "javascript":
                suffix = ".js"
                command = ["node"]
            elif language == "bash":
                suffix = ".sh"
                command = ["bash"]
            else:
                return AgentResult.error_result(
                    f"Unsupported language: {language}",
                    0.0
                )
            
            temp_file = self.temp_dir / f"exec_{id(code)}{suffix}"
            
            # Write code to file
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Make executable if shell script
            if language == "bash":
                temp_file.chmod(0o755)
            
            # Execute code
            process = await asyncio.create_subprocess_exec(
                *command, str(temp_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                # Clean up
                temp_file.unlink(missing_ok=True)
                
                if process.returncode == 0:
                    return AgentResult.success_result(
                        data={
                            "stdout": stdout.decode(),
                            "stderr": stderr.decode(),
                            "return_code": process.returncode
                        },
                        duration=0.0,
                        metadata={
                            "language": language,
                            "temp_file": str(temp_file)
                        }
                    )
                else:
                    return AgentResult.error_result(
                        f"Code execution failed with return code {process.returncode}: {stderr.decode()}",
                        0.0
                    )
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                temp_file.unlink(missing_ok=True)
                return AgentResult.error_result(
                    f"Code execution timed out after {timeout} seconds",
                    0.0
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _analyze_file(
        self,
        file_path: str,
        analysis_type: str = "general",
        **kwargs
    ) -> AgentResult:
        """Analyze file using local tools."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return AgentResult.error_result(
                    f"File not found: {file_path}",
                    0.0
                )
            
            analysis = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix,
                "analysis_type": analysis_type
            }
            
            # Basic file analysis
            if analysis_type == "general":
                # Get file info
                stat = file_path.stat()
                analysis.update({
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:]
                })
                
                # Try to read content if text file
                if file_path.suffix in [".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            analysis.update({
                                "content_preview": content[:500],
                                "line_count": len(content.splitlines()),
                                "char_count": len(content)
                            })
                    except UnicodeDecodeError:
                        analysis["content_type"] = "binary"
            
            elif analysis_type == "code":
                # Code-specific analysis
                if file_path.suffix == ".py":
                    # Python analysis using ast
                    try:
                        import ast
                        with open(file_path, "r") as f:
                            tree = ast.parse(f.read())
                        
                        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                        
                        analysis.update({
                            "functions": functions,
                            "classes": classes,
                            "function_count": len(functions),
                            "class_count": len(classes)
                        })
                    except Exception as e:
                        analysis["parse_error"] = str(e)
            
            return AgentResult.success_result(
                data=analysis,
                duration=0.0,
                metadata={"analysis_type": analysis_type}
            )
            
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _process_image(
        self,
        image_path: str,
        operation: str = "analyze",
        **kwargs
    ) -> AgentResult:
        """Process image using local tools."""
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                return AgentResult.error_result(
                    f"Image not found: {image_path}",
                    0.0
                )
            
            # Check if we have PIL/Pillow available
            try:
                from PIL import Image, ExifTags
                
                with Image.open(image_path) as img:
                    analysis = {
                        "image_path": str(image_path),
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                        "width": img.width,
                        "height": img.height
                    }
                    
                    # Get EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        if exif:
                            exif_data = {}
                            for tag_id, value in exif.items():
                                tag = ExifTags.TAGS.get(tag_id, tag_id)
                                exif_data[tag] = value
                            analysis["exif"] = exif_data
                    
                    # Perform operation
                    if operation == "resize":
                        new_size = kwargs.get("size", (800, 600))
                        output_path = self.temp_dir / f"resized_{image_path.name}"
                        
                        resized = img.resize(new_size)
                        resized.save(output_path)
                        
                        analysis["operation"] = "resize"
                        analysis["output_path"] = str(output_path)
                        analysis["new_size"] = new_size
                    
                    elif operation == "convert":
                        new_format = kwargs.get("format", "PNG")
                        output_path = self.temp_dir / f"converted_{image_path.stem}.{new_format.lower()}"
                        
                        img.save(output_path, format=new_format)
                        
                        analysis["operation"] = "convert"
                        analysis["output_path"] = str(output_path)
                        analysis["new_format"] = new_format
                
                return AgentResult.success_result(
                    data=analysis,
                    duration=0.0,
                    metadata={"operation": operation}
                )
                
            except ImportError:
                # Fallback to basic file analysis
                return await self._analyze_file(str(image_path), "general")
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _health_check(self) -> None:
        """Perform local agent health check."""
        try:
            # Test basic command execution
            process = await asyncio.create_subprocess_exec(
                "echo", "health_check",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise AgentError(f"Health check failed: {stderr.decode()}")
                
            if stdout.decode().strip() != "health_check":
                raise AgentError("Health check output mismatch")
                
        except Exception as e:
            raise AgentError(f"Health check failed: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)